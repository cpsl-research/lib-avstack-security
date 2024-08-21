from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from avstack.environment.objects import ObjectState
    from .propagation import AdvPropagator

from copy import deepcopy

from avstack.modules.perception.detections import BoxDetection
from avstack.modules.tracking import BasicBoxTrack3D


class TargetObject:
    def __init__(self, obj_state: "ObjectState"):
        self.obj_state = obj_state
        self.target_state = deepcopy(obj_state)
        self.last_position = obj_state.position
        self.timestamp = obj_state.timestamp
        self.propagation_model = None

    @property
    def t(self):
        return self.timestamp

    def set_propagation_model(self, model: "AdvPropagator"):
        self.propagation_model = model

    def as_track(self):
        """Format the target state as a track state"""
        return BasicBoxTrack3D(
            t0=self.obj_state.t,
            box3d=self.obj_state.box,
            reference=self.obj_state.reference,
            obj_type=self.obj_state.obj_type,
            v=self.obj_state.velocity.x,
        )

    def as_detection(self):
        """Format the target state as a detection"""
        return BoxDetection(
            source_identifier=0,
            box=self.obj_state.box,
            reference=self.obj_state.reference,
            obj_type=self.obj_state.obj_type,
            score=1.0,
        )

    def propagate(self, dt: float):
        """Updates target state with kinematics"""
        if self.propagation_model is None:
            raise RuntimeError("Need to initialize propagation model")
        self.propagation_model.propagate(dt, self.target_state)
        self.timestamp += dt
