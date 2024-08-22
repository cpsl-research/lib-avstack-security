from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from avstack.environment.objects import ObjectState
    from .propagation import AdvPropagator

from copy import deepcopy

from avstack.modules.perception.detections import BoxDetection
from avstack.modules.tracking import BasicBoxTrack3D


class TargetObject:
    def __init__(self, obj_state: "ObjectState"):
        self._obj_state = obj_state
        self._target_state = deepcopy(obj_state)
        self._propagation_model = None
        self._states_initialized = False
        self.last_position = obj_state.position
        self.timestamp = obj_state.timestamp

    @property
    def t(self) -> float:
        return self.timestamp

    def set_propagation_model(self, model: "AdvPropagator"):
        self._propagation_model = model

    def as_track(self) -> BasicBoxTrack3D:
        """Format the target state as a track state"""
        return BasicBoxTrack3D(
            t0=self._target_state.t,
            box3d=self._target_state.box,
            reference=self._target_state.reference,
            obj_type=self._target_state.obj_type,
            v=self._target_state.velocity.x,
        )

    def as_detection(self) -> BoxDetection:
        """Format the target state as a detection"""
        return BoxDetection(
            source_identifier=0,
            box=self._target_state.box,
            reference=self._target_state.reference,
            obj_type=self._target_state.obj_type,
            score=1.0,
        )

    def as_object_state(self) -> "ObjectState":
        return self._target_state

    def propagate(self, dt: float):
        """Updates target state with kinematics"""
        if self._propagation_model is None:
            raise RuntimeError("Need to initialize propagation model")
        initialize = not self._states_initialized
        self._propagation_model.propagate(dt, self._target_state, initialize=initialize)
        self.timestamp += dt
        self._states_initialized = True
