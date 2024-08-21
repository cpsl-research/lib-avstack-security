import numpy as np
from avstack.datastructs import DataContainer
from avstack.environment.objects import ObjectState
from avstack.geometry import Attitude, Box3D, GlobalOrigin3D, Position, Velocity

from avsec.multi_agent.adversary import AdversaryModel
from avsec.multi_agent.manifest import (
    FalseNegativeManifest,
    FalsePositiveManifest,
    TranslationManifest,
)
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
        obj.velocity = Velocity(np.random.randn(3), GlobalOrigin3D)
        obj.box = Box3D(position=obj.position, attitude=obj.attitude, hwl=[2, 2, 4])
        objs.append(obj)
    return objs


def test_adversary_with_manifests_and_props():
    ref_agent = GlobalOrigin3D
    dt_init = 1.0
    dt = 0.1
    n_frames = 20
    manifests = [
        FalsePositiveManifest(fp_poisson=8),
        FalseNegativeManifest(fn_fraction=0.5),
        TranslationManifest(tr_fraction=0.5),
    ]
    propagators = [
        StaticPropagator(),
        MarkovPropagator(),
        TrajectoryPropagator(dx_total=np.array([10, 0, 0])),
    ]
    for manifest in manifests:
        for propagator in propagators:
            # initialize adversary
            adversary = AdversaryModel(
                dt_init=dt_init,
                propagator=propagator,
                manifest_fp=manifest
                if "positive" in str(manifest.__class__).lower()
                else None,
                manifest_fn=manifest
                if "negative" in str(manifest.__class__).lower()
                else None,
                manifest_tr=manifest
                if "translation" in str(manifest.__class__).lower()
                else None,
            )
            # run over frames
            for i_frame in range(n_frames):
                timestamp = i_frame * dt

                # construct random objects
                n_objs_fixed = (
                    0 if "positive" in str(manifest.__class__).lower() else 10
                )
                objects = DataContainer(
                    frame=i_frame,
                    timestamp=timestamp,
                    source_identifier="",
                    data=random_objects(n_objects=n_objs_fixed),
                )

                # run the adversary model
                objects = adversary(objects, ref_agent)

                # check the number of objects
                if timestamp < dt_init:
                    assert len(objects) == n_objs_fixed
                else:
                    if "positive" in str(manifest.__class__).lower():
                        assert len(objects) > n_objs_fixed
                    elif "negative" in str(manifest.__class__).lower():
                        assert len(objects) < n_objs_fixed
                    else:
                        assert len(objects) == n_objs_fixed

            # check closeness of outcomes
            for target in adversary.targets["false_positive"]:
                if "static" in str(propagator.__class__).lower():
                    assert np.allclose(
                        target.last_position.x, target.target_state.position.x
                    )
                else:
                    assert not np.allclose(
                        target.last_position.x, target.target_state.position.x
                    )


def test_adv_translation():
    ref_agent = GlobalOrigin3D
    dt_init = 1.0
    dt = 0.1
    n_frames = 20
    adversary = AdversaryModel(
        propagator=StaticPropagator(),
        manifest_tr=TranslationManifest(tr_fraction=0.5),
        dt_init=dt_init,
    )
    # run over frames
    for i_frame in range(n_frames):
        timestamp = i_frame * dt

        # construct random objects
        n_objs_fixed = 10
        objects = DataContainer(
            frame=i_frame,
            timestamp=timestamp,
            source_identifier="",
            data=random_objects(n_objects=n_objs_fixed),
        )

        # run the adversary model
        objects = adversary(objects, ref_agent)
