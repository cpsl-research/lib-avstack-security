from typing import TYPE_CHECKING, Union


if TYPE_CHECKING:
    from avstack.geometry import ReferenceFrame, Shape
    from .manifest import AdvManifest
    from .propagation import AdvPropagator

import numpy as np
from avstack.config import ConfigDict
from avstack.datastructs import DataContainer
from avstack.environment.objects import ObjectState
from avstack.modules.perception.detections import BoxDetection

from avsec.config import AVSEC

from .manifest import FalseNegativeManifest, FalsePositiveManifest, TranslationManifest


@AVSEC.register_module()
class AdversaryModel:
    def __init__(
        self,
        propagator: "AdvPropagator",
        manifest_fp: Union[ConfigDict, "FalsePositiveManifest", None] = None,
        manifest_fn: Union[ConfigDict, "FalseNegativeManifest", None] = None,
        manifest_tr: Union[ConfigDict, "TranslationManifest", None] = None,
        manifest: Union[ConfigDict, "AdvManifest", None] = None,
        dt_init: float = 2.0,
        dt_reset: float = 10.0,
        seed: Union[int, None] = None,
    ):
        # parse the manifest
        if manifest is not None:
            if any(
                [man is not None for man in [manifest_fp, manifest_fn, manifest_tr]]
            ):
                raise ValueError(
                    "If manifest is passed, do not pass other manifest_TYPE"
                )
            manifest = (
                AVSEC.build(manifest, default_args={"seed": seed})
                if isinstance(manifest, dict)
                else manifest
            )
            self.manifest_fp = None
            self.manifest_fn = None
            self.manifest_tr = None
            if isinstance(manifest, FalsePositiveManifest):
                self.manifest_fp = manifest
            elif isinstance(
                manifest, TranslationManifest
            ):  # must be b4 FN due to inheritance
                self.manifest_tr = manifest
            elif isinstance(manifest, FalseNegativeManifest):
                self.manifest_fn = manifest
            else:
                raise NotImplementedError(type(manifest))
        else:
            self.manifest_fp = (
                AVSEC.build(manifest_fp, default_args={"seed": seed})
                if isinstance(manifest_fp, dict)
                else manifest_fp
            )
            self.manifest_fn = (
                AVSEC.build(manifest_fn, default_args={"seed": seed})
                if isinstance(manifest_fn, dict)
                else manifest_fn
            )
            self.manifest_tr = (
                AVSEC.build(manifest_tr, default_args={"seed": seed})
                if isinstance(manifest_tr, dict)
                else manifest_tr
            )

        # parse the propagator
        self.propagator = (
            AVSEC.build(propagator, default_args={"seed": seed})
            if isinstance(propagator, dict)
            else propagator
        )

        # set other inputs
        self.dt_init = dt_init
        self.dt_reset = dt_reset
        self.reset()

    def reset(self):
        self._t_start = None
        self.targets_initialized = False
        self.targets = {"false_positive": [], "false_negative": [], "translations": []}

    def __call__(
        self,
        objects: "DataContainer",
        fov: "Shape",
        reference: "ReferenceFrame",
        fn_dist_threshold: int = 6,
        threshold_obj_dist: float = 70.0,
    ) -> "DataContainer":
        """Processing the targets for the attacks

        Args:
            objects: a datacontainer of objects or detections
            reference: the reference frame of the agent

        Returns:
            obj_convert: a datacontainer of detections

        False positives:
        if we have already created a list of false positives, just propagate
        those in time and append to the list of existing objects

        False negatives:
        find if there is a track with the same ID as identified before or one
        that is sufficiently close in space to the target and eliminate
        it from the outgoing message.
        """
        # handle timing
        timestamp = objects.timestamp
        if self._t_start is None:
            self._t_start = timestamp
        elif (timestamp - self._t_start) > self.dt_reset:
            self.reset()
            return objects, fov

        # ===========================================
        # Data type conversions
        # ===========================================

        def detection_to_obj(det: "BoxDetection", timestamp: float) -> "ObjectState":
            obj = ObjectState(det.obj_type, score=det.score)
            obj.set(
                t=timestamp,
                position=det.box.position,
                attitude=det.box.attitude,
                box=det.box,
            )
            return obj

        # convert types to objects
        objs_as_states = [isinstance(obj, ObjectState) for obj in objects]
        objs_as_dets = [isinstance(obj, BoxDetection) for obj in objects]

        if all(objs_as_states):
            input_as_objects = objects
            output_type = "ObjectState"
        elif all(objs_as_dets):
            input_as_objects = objects.apply_and_return(
                detection_to_obj, timestamp=timestamp
            )
            output_type = "BoxDetection"
        else:
            raise RuntimeError("Non-uniform object types")

        # ===========================================
        # Initialization
        # ===========================================

        if not self.targets_initialized:
            if (timestamp - self._t_start) >= self.dt_init:
                self.initialize_uncoordinated(
                    objects=input_as_objects, reference=reference
                )

        # ===========================================
        # Run model
        # ===========================================

        if self.targets_initialized:
            # process false positives
            for obj_fp in self.targets["false_positive"]:
                obj_fp.change_reference(reference, inplace=True)
                obj_fp.propagate(dt=(timestamp - obj_fp.t))
                input_as_objects.append(obj_fp.as_object_state())

            # process false negatives
            for obj_fn in self.targets["false_negative"]:
                obj_fn.change_reference(reference, inplace=True)
                # perform assignment of existing detections/tracks to targets
                dists = [
                    obj.position.distance(obj_fn.last_position, check_reference=False)
                    for obj in input_as_objects
                ]
                idx_select = np.argmin(dists)
                if dists[idx_select] <= fn_dist_threshold:
                    obj_fn.last_position = input_as_objects[idx_select].position
                    # remove the ones that were assigned
                    del input_as_objects[idx_select]

            # process translations
            for obj_tr in self.targets["translations"]:
                obj_tr.change_reference(reference, inplace=True)
                # perform assignment of existing detections/tracks to obj states
                dists = [
                    obj.position.distance(obj_tr.last_position, check_reference=False)
                    for obj in input_as_objects
                ]
                idx_select = np.argmin(dists)
                if dists[idx_select] <= fn_dist_threshold:
                    obj_tr.last_position = input_as_objects[idx_select].position
                    # translate the ones that were assigned
                    obj_tr.propagate(dt=(timestamp - obj_tr.t))
                    input_as_objects[idx_select] = obj_tr.as_object_state()

            # filter objects outside a distance
            input_as_objects = input_as_objects.filter(
                lambda obj: obj.position.norm() < threshold_obj_dist
            )

        # ===========================================
        # Convert outputs
        # ===========================================

        # define conversion function
        def obj_to_detection(obj: "ObjectState"):
            return BoxDetection(
                source_identifier="",
                box=obj.box,
                reference=obj.reference,
                obj_type=obj.obj_type,
                score=obj.score,
            )

        # run conversion
        if output_type == "ObjectState":
            outputs = input_as_objects
        elif output_type == "BoxDetection":
            outputs = input_as_objects.apply_and_return(obj_to_detection)
        else:
            raise NotImplementedError(output_type)

        return outputs, fov

    def initialize_uncoordinated(
        self, objects: "DataContainer", reference: "ReferenceFrame"
    ):
        """Initialize uncoordinated attack by selecting attack targets

        Args:
            objects: existing objects
            reference: reference frame for agent
        """
        # select false positive objects randomly in space
        if self.manifest_fp is not None:
            self.targets["false_positive"] = self.manifest_fp.select(
                timestamp=objects.timestamp,
                reference=reference,
            )

        # select false negative targets from existing objects
        if self.manifest_fn is not None:
            self.targets["false_negative"] = self.manifest_fn.select(
                objects=objects,
            )

        # select translation targets from existing objects
        if self.manifest_tr is not None:
            self.targets["translations"] = self.manifest_tr.select(objects=objects)

        # set the propagation model
        for vs in self.targets.values():
            for v in vs:
                v.set_propagation_model(self.propagator)

        # set fields
        self.targets_initialized = True
