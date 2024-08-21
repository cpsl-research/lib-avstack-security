from typing import TYPE_CHECKING, Union


if TYPE_CHECKING:
    from avstack.datastructs import DataContainer
    from avstack.geometry import ReferenceFrame
    from .manifest import (
        FalseNegativeManifest,
        FalsePositiveManifest,
        TranslationManifest,
    )
    from .propagation import AdvPropagator

import numpy as np
from avstack.datastructs import PriorityQueue


class AdversaryModel:
    def __init__(
        self,
        propagator: "AdvPropagator",
        manifest_fp: Union["FalsePositiveManifest", None] = None,
        manifest_fn: Union["FalseNegativeManifest", None] = None,
        manifest_tr: Union["TranslationManifest", None] = None,
        dt_init: float = 2.0,
    ):
        self.manifest_fp = manifest_fp
        self.manifest_fn = manifest_fn
        self.manifest_tr = manifest_tr
        self.propagator = propagator
        self.dt_init = dt_init
        self.reset()

    def reset(self):
        self._t_start = None
        self.data_buffer = PriorityQueue(max_size=5, max_heap=True)
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
        # add to the data buffer
        timestamp = objects.timestamp
        if self._t_start is None:
            self._t_start = timestamp
        self.data_buffer.push(
            priority=timestamp,
            item=objects,
        )

        # run initialization logic
        if not self.targets_initialized:
            if (timestamp - self._t_start) >= self.dt_init:
                self.initialize_uncoordinated(
                    objects=objects, reference_agent=reference_agent
                )

        if self.targets_initialized:
            # process false positives
            for obj_fp in self.targets["false_positive"]:
                obj_fp.propagate(dt=(timestamp - obj_fp.t))
                obj_fp_convert = obj_fp.as_detection()
                objects.append(obj_fp_convert)

            # process false negatives
            for obj_fn in self.targets["false_negative"]:
                # perform assignment of existing detections/tracks to targets
                dists = [
                    obj.position.distance(obj_fn.last_position, check_reference=False)
                    for obj in objects
                ]
                idx_select = np.argmin(dists)
                if dists[idx_select] <= fn_dist_threshold:
                    obj_fn.last_position = objects[idx_select].position
                    # remove the ones that were assigned
                    del objects[idx_select]

            # process translations
            for obj_tr in self.targets["translations"]:
                # perform assignment of existing detections/tracks to obj states
                dists = [
                    obj.position.distance(obj_tr.last_position, check_reference=False)
                    for obj in objects
                ]
                idx_select = np.argmin(dists)
                if dists[idx_select] <= fn_dist_threshold:
                    obj_tr.last_position = objects[idx_select].position
                    # translate the ones that were assigned
                    obj_tr.propagate(dt=(timestamp - obj_fn.t))
                    obj_tr_convert = obj_tr.as_detection()
                    objects.append(obj_tr_convert)

            # filter objects outside a distance
            objects = objects.filter(
                lambda obj: obj.position.norm() < threshold_obj_dist
            )

        return objects

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
            self.targets["translation"] = self.manifest_tr.select(objects=objects)

        # set the propagation model
        for vs in self.targets.values():
            for v in vs:
                v.set_propagation_model(self.propagator)

        # set fields
        self.targets_initialized = True
