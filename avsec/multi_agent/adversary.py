from typing import TYPE_CHECKING, List


if TYPE_CHECKING:
    from avstack.datastructs import DataContainer
    from avstack.geometry import Position

import numpy as np
from avstack.datastructs import PriorityQueue

from avsec.multi_agent.selection import select_false_negatives, select_false_positives


class AdversaryModel:
    def __init__(self, coordinated: bool, dt_init: float = 2.0):
        self.coordinated = coordinated
        self.dt_init = dt_init
        self.reset()

    def reset(self):
        self._t_start = None
        self.data_buffer = PriorityQueue(max_size=5, max_heap=True)
        self.ready = False
        self.targets_initialized = False
        self.targets = {"false_positive": [], "false_negative": [], "translations": []}

    def __call__(
        self,
        objects: "DataContainer",
        fn_threshold: int = 6,
        threshold_obj_dist: float = 70.0,
    ):
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
        if (not self.coordinated) and (not self.targets_initialized):
            if (timestamp - self._t_start) >= self.dt_init:
                self.initialize_uncoordinated(objects=objects)

        # process false positives
        for obj_fp in self.targets["false_positive"]:
            obj_fp.propagate(dt=(timestamp - obj_fp.t))
            if self.coordinated:
                obj_fp_convert = obj_fp.as_track()
            else:
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
            if dists[idx_select] <= fn_threshold:
                # remove the ones that were assigned
                obj_fn.last_position = objects[idx_select].position
                del objects[idx_select]

        # filter objects outside a distance
        objects = [
            objects for obj in objects if obj.position.norm() < threshold_obj_dist
        ]

        return objects

    def initialize_coordinated(
        self, fp_directive: List["Position"], fn_directive: List["Position"]
    ):
        """Called when receiving a directive from the coordinating adversary

        Args:
            fp_directive - List[BasicBoxTrack3D]: Information on false positives from coordinator
            fn_directive - List[BasicBoxTrack3D]: Information on false negatiaves from coordinator
        """
        self.reset_targets()
        self.ready = True
        self.init_targets = True

        # if not len(msg.track_arrays) == 2:
        #     raise ValueError("Input must be of length 2 -- FP and FN")
        # else:
        #     to_frame = self.get_parameter("attack_agent_name").value
        #     for obj_fp in msg.track_arrays[0].tracks:
        #         obj_fp_stamped = BoxTrack(header=msg.header, track=obj_fp)
        #         obj_fp_in_agent_frame = self._tf_buffer.transform(
        #             object_stamped=obj_fp_stamped,
        #             target_frame=to_frame,
        #         )
        #         obj_fp_avstack = TrackBridge.boxtrack_to_avstack(obj_fp_in_agent_frame)
        #         self.targets["false_positive"].append(
        #             TargetObject(obj_state=obj_fp_avstack)
        #         )

        # fn targets are in the world coordinate frame -- convert to agent
        # for obj_fn in msg.track_arrays[1].tracks:
        #     obj_fn_in_agent_frame = do_transform_boxtrack(obj_fn, tf_to_agent)
        #     obj_fn_avstack = TrackBridge.boxtrack_to_avstack(
        #         obj_fn_in_agent_frame, header=tf_to_agent.header
        #     )
        #     self.targets["false_negative"].append(TargetObject(obj_state=obj_fn_avstack))

        raise NotImplementedError()

    def initialize_uncoordinated(self, objects: "DataContainer"):
        """Initialize uncoordinated attack by selecting attack targets

        Args:
            objects: existing objects
        """
        # select false positive objects randomly in space
        self.targets["false_positive"] = select_false_positives(
            fp_poisson=self.fp_poisson_uncoord,
            reference_agent=None,
            reference=None,
            x_sigma=30,
            v_sigma=10,
            hwl=[2, 2, 4],
        )

        # select false negative targets from existing objects
        self.targets["false_negative"] = select_false_negatives(
            existing_objects=objects,
            fn_fraction=self.fn_fraction_uncoord,
        )

        # select translation targets from existing objects
        self.targets["translation"] = []

        # set fields
        self.targets_initialized = True
