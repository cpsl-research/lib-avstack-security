from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from avstack.environment.objects import ObjectState
    from avstack.geometry import ReferenceFrame

import numpy as np
from avstack.geometry import Attitude, Velocity, transform_orientation


def initialize_velocity_attitude(
    reference_agent: "ReferenceFrame",
    v_sigma: float = 10,
    dx_total: np.ndarray = None,
    dt_total: np.ndarray = None,
):
    """Initialize a random velocity/attitude in ground projected frame"""

    # velocity is random in x-y
    if dx_total is None:
        v_vec = v_sigma * np.array([np.random.randn(), np.random.randn(), 0])
    else:
        v_vec = dx_total / dt_total

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
    return velocity, attitude


class AdvPropagator:
    def propagate(self, dt: float, obj: "ObjectState"):
        self._propagate(dt, obj)
        obj.t += dt

    def _propagate(self, dt: float, obj: "ObjectState"):
        raise NotImplementedError


class StaticPropagator(AdvPropagator):
    def _propagate(self, dt: float, obj: "ObjectState"):
        """Static propagation is nothing"""
        pass


class MarkovPropagator(AdvPropagator):
    def __init__(self, v_sigma: float = 10, dv_sigma: float = 1):
        self.v_sigma = v_sigma
        self.dv_sigma = dv_sigma

    def _propagate(self, dt: float, obj: "ObjectState"):
        """Apply a markov model to velocity and pass to position"""
        # initialize velocity and attitude, if needed
        if obj.velocity is None:
            obj.velocity, obj.attitude = initialize_velocity_attitude(
                v_sigma=self.v_sigma, reference_agent=obj.reference
            )

        # propagate with kinematics
        obj.position = obj.position + obj.velocity.x * dt
        if obj.box is not None:
            obj.box.position = obj.position
            obj.box.attitude = obj.attitude


class TrajectoryPropagator(AdvPropagator):
    def __init__(self, dx_total: np.ndarray, dt_total: float = 10):
        """Propagate an object to a point in spaces

        Args:
            dx_total: the change in position to achieve
            dt_total: the time over which to achieve the dx
        """
        self.dx_total = dx_total
        self.dt_total = dt_total
        self.dt_elapsed = 0
        self._v = dx_total / dt_total

    def _propagate(self, dt: float, obj: "ObjectState"):
        # initialize velocity and attitude, if needed
        if obj.velocity is None:
            obj.velocity, obj.attitude = initialize_velocity_attitude(
                dx_total=self.dx_total,
                dt_total=self.dt_total,
                reference_agent=obj.reference
            )

        # propagate along trajectory
        obj.position = obj.position + obj.velocity.x * dt
        if obj.box is not None:
            obj.box.position = obj.position
            obj.box.attitude = obj.attitude