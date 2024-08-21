from typing import TYPE_CHECKING, Union


if TYPE_CHECKING:
    from avstack.geometry import ReferenceFrame
    from .manifest import (
        FalseNegativeManifest,
        FalsePositiveManifest,
        TranslationManifest,
    )
    from .propagation import AdvPropagator

import numpy as np
from avstack.datastructs import DataContainer
from avstack.environment.objects import ObjectState
from avstack.modules.perception.detections import BoxDetection


class AdversaryModel:
    def __init__(
        self,
        propagator: "AdvPropagator",
        manifest_fp: Union["FalsePositiveManifest", None] = None,
        manifest_fn: Union["FalseNegativeManifest", None] = None,
        manifest_tr: Union["TranslationManifest", None] = None,
        dt_init: float = 2.0,
        dt_reset: float = 10.0,
    ):
        self.manifest_fp = manifest_fp
        self.manifest_fn = manifest_fn
        self.manifest_tr = manifest_tr
        self.propagator = propagator
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
        reference_agent: "ReferenceFrame",
        fn_dist_threshold: int = 6,
        threshold_obj_dist: float = 70.0,
    ) -> "DataContainer":
        """Processing the targets for the attacks

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
            return objects

        # convert to objects
        obj_convert = DataContainer(
            frame=objects.frame,
            timestamp=objects.timestamp,
            source_identifier=objects.source_identifier,
            data=[],
        )
        for obj in objects:
            if isinstance(obj, ObjectState):
                obj_convert.append(obj)
            elif isinstance(obj, BoxDetection):
                new_obj = ObjectState(obj.obj_type)
                new_obj.set(
                    t=timestamp,
                    position=obj.box.position,
                    attitude=obj.box.attitude,
                    box=obj.box,
                )
                obj_convert.append(new_obj)
            else:
                raise NotImplementedError(type(obj))

        # run initialization logic
        if not self.targets_initialized:
            if (timestamp - self._t_start) >= self.dt_init:
                self.initialize_uncoordinated(
                    objects=obj_convert, reference_agent=reference_agent
                )

        if self.targets_initialized:
            # process false positives
            for obj_fp in self.targets["false_positive"]:
                obj_fp.propagate(dt=(timestamp - obj_fp.t))
                obj_fp_convert = obj_fp.as_detection()
                obj_convert.append(obj_fp_convert)

            # process false negatives
            for obj_fn in self.targets["false_negative"]:
                # perform assignment of existing detections/tracks to targets
                dists = [
                    obj.position.distance(obj_fn.last_position, check_reference=False)
                    for obj in obj_convert
                ]
                idx_select = np.argmin(dists)
                if dists[idx_select] <= fn_dist_threshold:
                    obj_fn.last_position = obj_convert[idx_select].position
                    # remove the ones that were assigned
                    del obj_convert[idx_select]

            # process translations
            for obj_tr in self.targets["translations"]:
                # perform assignment of existing detections/tracks to obj states
                dists = [
                    obj.position.distance(obj_tr.last_position, check_reference=False)
                    for obj in obj_convert
                ]
                idx_select = np.argmin(dists)
                if dists[idx_select] <= fn_dist_threshold:
                    obj_tr.last_position = obj_convert[idx_select].position
                    # translate the ones that were assigned
                    obj_tr.propagate(dt=(timestamp - obj_tr.t))
                    obj_tr_convert = obj_tr.as_detection()
                    obj_convert[idx_select] = obj_tr_convert

            # filter objects outside a distance
            obj_convert = obj_convert.filter(
                lambda obj: obj.position.norm() < threshold_obj_dist
            )

        return obj_convert

    def initialize_uncoordinated(
        self, objects: "DataContainer", reference_agent: "ReferenceFrame"
    ):
        """Initialize uncoordinated attack by selecting attack targets

        Args:
            objects: existing objects
            reference_agent: reference frame for agent
        """
        # select false positive objects randomly in space
        if self.manifest_fp is not None:
            self.targets["false_positive"] = self.manifest_fp.select(
                timestamp=objects.timestamp,
                reference_agent=reference_agent,
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
