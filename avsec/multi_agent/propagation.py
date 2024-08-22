from typing import TYPE_CHECKING, Union


if TYPE_CHECKING:
    from avstack.environment.objects import ObjectState
    from avstack.geometry import ReferenceFrame

import numpy as np
from avstack.geometry import Attitude, Velocity, transform_orientation

from avsec.config import AVSEC


class AdvPropagator:
    def __init__(self, seed: Union[int, None] = None):
        self.rng = np.random.RandomState(seed)

    @property
    def rng(self) -> np.random.RandomState:
        return self._rng if self._rng is not None else np.random

    @rng.setter
    def rng(self, rng):
        self._rng = rng

    def initialize_velocity_attitude(
        self,
        reference_agent: "ReferenceFrame",
        v_sigma: float = 10,
        v_max: Union[float, None] = None,
        dx_total: np.ndarray = None,
        dt_total: np.ndarray = None,
    ):
        """Initialize a random velocity/attitude in ground projected frame"""

        # velocity is random in x-y
        if dx_total is None:
            v_vec = v_sigma * np.array([self.rng.randn(), self.rng.randn(), 0])
        else:
            v_vec = dx_total / dt_total
        if v_max is not None:
            if np.linalg.norm(v_vec) > v_max:
                v_vec = v_max * v_vec / np.linalg.norm(v_vec)

        # attitude is in direction of velocity
        yaw = np.arctan2(v_vec[1], v_vec[0])
        euler = [0, 0, yaw]
        q_obj = transform_orientation(euler, "euler", "quat")

        # return to original reference frame
        reference_agent_gp = reference_agent.get_ground_projected_reference()
        velocity = Velocity(v_vec, reference=reference_agent_gp).change_reference(
            reference_agent, inplace=False
        )
        attitude = Attitude(q_obj, reference=reference_agent_gp).change_reference(
            reference_agent, inplace=False
        )

        # define the ground projected plane in two vectors
        plane = reference_agent.get_ground_projected_plane()

        return velocity, attitude, plane

    def propagate(self, dt: float, obj: "ObjectState", initialize: bool = False):
        self._propagate(dt, obj, initialize)
        obj.t += dt

    def _propagate(self, dt: float, obj: "ObjectState", initialize: bool):
        raise NotImplementedError


@AVSEC.register_module()
class StaticPropagator(AdvPropagator):
    def _propagate(self, dt: float, obj: "ObjectState", initialize: bool):
        """Static propagation is nothing but ref transformation"""


@AVSEC.register_module()
class MarkovPropagator(AdvPropagator):
    def __init__(self, v_sigma: float = 10, v_max: float = 30, dv_sigma: float = 0.50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v_sigma = v_sigma
        self.v_max = v_max
        self.dv_sigma = dv_sigma
        self.plane = []

    def _propagate(self, dt: float, obj: "ObjectState", initialize: bool):
        """Apply a markov model to velocity and pass to position"""
        # initialize velocity and attitude, if needed
        if (obj.velocity is None) or (initialize):
            obj.velocity, obj.attitude, self.plane = self.initialize_velocity_attitude(
                v_sigma=self.v_sigma, v_max=self.v_max, reference_agent=obj.reference
            )

        # add some noise to velocity in the plane
        dv_vec = (
            self.dv_sigma * self.rng.randn() * self.plane[0]
            + self.dv_sigma * self.rng.randn() * self.plane[1]
        )
        obj.velocity.x += dv_vec * dt

        # propagate with kinematics
        obj.position = obj.position + obj.velocity.x * dt
        if obj.box is not None:
            obj.box.position = obj.position
            obj.box.attitude = obj.attitude


@AVSEC.register_module()
class TrajectoryPropagator(AdvPropagator):
    def __init__(self, dx_total: np.ndarray, dt_total: float = 10, *args, **kwargs):
        """Propagate an object to a point in spaces

        Args:
            dx_total: the change in position to achieve
            dt_total: the time over which to achieve the dx
        """
        super().__init__(*args, **kwargs)
        self.dx_total = dx_total
        self.dt_total = dt_total
        self.dt_elapsed = 0
        self.plane = []

    def _propagate(self, dt: float, obj: "ObjectState", initialize: bool):
        # initialize velocity and attitude, if needed
        if (obj.velocity is None) or (initialize):
            obj.velocity, obj.attitude, self.plane = self.initialize_velocity_attitude(
                dx_total=self.dx_total,
                dt_total=self.dt_total,
                v_max=None,
                reference_agent=obj.reference,
            )

        # propagate along trajectory
        obj.position = obj.position + obj.velocity.x * dt
        if obj.box is not None:
            obj.box.position = obj.position
            obj.box.attitude = obj.attitude
