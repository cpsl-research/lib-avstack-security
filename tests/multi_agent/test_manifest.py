import numpy as np
from avstack.environment.objects import ObjectState
from avstack.geometry import GlobalOrigin3D, ReferenceFrame

from avsec.multi_agent.manifest import (
    FalseNegativeManifest,
    FalsePositiveManifest,
    TranslationManifest,
)


def test_false_positive_manifest():
    np.random.seed(1)
    manifest = FalsePositiveManifest(fp_poisson=20.0, min_select=1)
    targets = manifest.select(timestamp=1.0, reference=GlobalOrigin3D)
    assert len(targets) > 0


def test_false_negative_manifest():
    np.random.seed(1)
    manifest = FalseNegativeManifest(fn_poisson=2, min_select=1)
    objs = [ObjectState("car") for _ in range(10)]
    targets = manifest.select(objs)
    assert len(targets) > 0


def test_translation_manifest():
    np.random.seed(1)
    manifest = TranslationManifest(tr_poisson=2, min_select=1)
    objs = [ObjectState("car") for _ in range(10)]
    targets = manifest.select(objs)
    assert len(targets) > 0


def test_manifest_ground_plane():
    np.random.seed(1)
    agent_reference = ReferenceFrame(
        x=np.random.randn(3), q=np.quaternion(1), reference=GlobalOrigin3D
    )
    lidar_reference = ReferenceFrame(
        x=np.array([0, 0, 2]), q=np.quaternion(1), reference=agent_reference
    )
    manifest = FalsePositiveManifest(fp_poisson=20.0)
    targets = manifest.select(timestamp=1.0, reference=lidar_reference)
    for target in targets:
        assert np.isclose(
            0,
            target.as_object_state()
            .position.change_reference(GlobalOrigin3D, inplace=False)
            .x[2],
        )
