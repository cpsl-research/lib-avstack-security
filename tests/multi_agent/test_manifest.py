import numpy as np
from avstack.environment.objects import ObjectState
from avstack.geometry import GlobalOrigin3D

from avsec.multi_agent.manifest import (
    FalseNegativeManifest,
    FalsePositiveManifest,
    TranslationManifest,
)


def test_false_positive_manifest():
    np.random.seed(1)
    manifest = FalsePositiveManifest(fp_poisson=2.0)
    targets = manifest.select(timestamp=1.0, reference_agent=GlobalOrigin3D)
    assert len(targets) > 0


def test_false_negative_manifest():
    np.random.seed(1)
    manifest = FalseNegativeManifest(fn_fraction=0.5)
    objs = [ObjectState("car") for _ in range(10)]
    targets = manifest.select(objs)
    assert len(targets) > 0


def test_translation_manifest():
    np.random.seed(1)
    manifest = TranslationManifest(tr_fraction=0.5)
    objs = [ObjectState("car") for _ in range(10)]
    targets = manifest.select(objs)
    assert len(targets) > 0
