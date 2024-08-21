import numpy as np
from avstack.environment.objects import ObjectState
from avstack.geometry import GlobalOrigin3D, Position

from avsec.multi_agent.propagation import MarkovPropagator
from avsec.multi_agent.target import TargetObject


def random_objects(n_objects: int = 10):
    objs = []
    for i in range(n_objects):
        obj = ObjectState("car", ID=i)
        obj.t = 0.0
        obj.position = Position(np.random.randn(3), GlobalOrigin3D)
        objs.append(obj)
    return objs


def test_target_propagation():
    obj = random_objects(1)[0]
    target = TargetObject(obj_state=obj)

    # without prop set will fail
    try:
        target.propagate(dt=1.0)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Shouldn't get here")

    # with prop set
    target.set_propagation_model(MarkovPropagator())
    target.propagate(dt=1.0)
    assert target.target_state.timestamp == 1.0
