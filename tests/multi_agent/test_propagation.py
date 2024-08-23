import numpy as np
from avstack.environment.objects import ObjectState
from avstack.geometry import Attitude, Box3D, GlobalOrigin3D, Position

from avsec.multi_agent.propagation import (
    MarkovPropagator,
    StaticPropagator,
    TrajectoryPropagator,
)


def random_objects(n_objects: int = 10, seed: int = None):
    np.random.seed(seed)
    objs = []
    for i in range(n_objects):
        obj = ObjectState("car", ID=i)
        obj.t = 0.0
        obj.position = Position(np.random.randn(3), GlobalOrigin3D)
        obj.attitude = Attitude(np.quaternion(1), GlobalOrigin3D)
        # obj.velocity = Velocity(np.random.randn(3), GlobalOrigin3D)
        obj.box = Box3D(position=obj.position, attitude=obj.attitude, hwl=[2, 2, 4])
        objs.append(obj)
    return objs


def test_static_propagator():
    propagator = StaticPropagator()
    objs = random_objects()
    for obj in objs:
        x_before = obj.position.x.copy()
        propagator.propagate(dt=1.0, obj=obj)
        assert np.allclose(x_before, obj.position.x)


def test_markov_propagator():
    propagator = MarkovPropagator()
    objs = random_objects()
    for obj in objs:
        x_before = obj.position.x.copy()
        propagator.propagate(dt=1.0, obj=obj)
        assert not np.allclose(x_before, obj.position.x)


def test_trajectory_propagator():
    propagator = TrajectoryPropagator(dx_total=np.array([10, 0, 0]))
    objs = random_objects()
    for obj in objs:
        x_before = obj.position.x.copy()
        propagator.propagate(dt=1.0, obj=obj)
        assert x_before[0] != obj.position.x[0]
        assert x_before[1] == obj.position.x[1]
        assert x_before[2] == obj.position.x[2]


def test_trajectory_propagator_range():
    propagator = TrajectoryPropagator(dr_total=20)
    objs = random_objects()
    for obj in objs:
        x_before = obj.position.x.copy()
        propagator.propagate(dt=1.0, obj=obj)
        assert x_before[0] != obj.position.x[0]
        assert x_before[1] != obj.position.x[1]
        assert x_before[2] == obj.position.x[2]
