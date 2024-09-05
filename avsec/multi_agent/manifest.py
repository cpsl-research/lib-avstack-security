from typing import TYPE_CHECKING, List, Union


if TYPE_CHECKING:
    from avstack.datastructs import DataContainer
    from avstack.geometry import ReferenceFrame

import numpy as np
from avstack.environment.objects import ObjectState
from avstack.geometry import Attitude, Box3D, Position

from avsec.config import AVSEC

from .target import TargetObject


class AdvManifest:
    def __init__(
        self,
        seed: Union[int, None] = None,
        n_select_poisson: Union[float, None] = None,
        exact_select: Union[int, None] = None,
        min_select: int = 0,
        max_select: int = np.inf,
    ):
        self.rng = np.random.RandomState(seed)
        self.n_select_poisson = n_select_poisson
        self.min_select = min_select if exact_select is None else exact_select
        self.max_select = max_select if exact_select is None else exact_select

    @property
    def rng(self) -> np.random.RandomState:
        return self._rng if self._rng is not None else np.random

    @rng.setter
    def rng(self, rng):
        self._rng = rng

    def select(self):
        raise NotImplementedError


@AVSEC.register_module()
class FalsePositiveManifest(AdvManifest):
    def __init__(
        self,
        x_sigma: float = 15,
        x_bias: float = 0,
        hwl: List[float] = [2, 2, 4],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.x_sigma = np.asarray(x_sigma)
        self.x_bias = np.asarray(x_bias)
        self.hwl = hwl

    def select(
        self,
        timestamp: float,
        reference: "ReferenceFrame",
        *args,
        **kwargs,
    ) -> List[TargetObject]:
        """Create some false positives for attacks.

        Keep in mind the reference frame may not be parallel to the group plane.
        Therefore, the generation of false positive targets should be assumed
        to be made relative to a projected sensor plane that is co-planar with ground.
        """
        reference_gp = reference.get_ground_projected_reference()

        # sample the number of false positives
        n_fp_poisson = (
            np.round(self.rng.poisson(self.n_select_poisson))
            if self.n_select_poisson is not None
            else 0
        )
        n_fp = max(self.min_select, min(self.max_select, n_fp_poisson))

        # construct target positions
        targets = []
        for i in range(n_fp):
            # get data -- flat on ground plane
            x_vec = self.x_bias + self.x_sigma * np.array(
                [self.rng.randn(), self.rng.randn(), 0]
            )
            q_vec = np.quaternion(1)  # identity until velocity set later

            # adjust for false positive selection that is coplanar with ground
            pos = Position(x_vec, reference=reference_gp).change_reference(
                reference, inplace=False
            )
            att = Attitude(q_vec, reference=reference_gp).change_reference(
                reference, inplace=False
            )

            # populate object state
            obj = ObjectState(ID=i, obj_type="car")
            obj.set(
                t=timestamp,
                position=pos,
                box=Box3D(pos, att, self.hwl, where_is_t="bottom"),
                attitude=att,
            )
            targets.append(TargetObject(obj_state=obj))
        return targets


@AVSEC.register_module()
class FalseNegativeManifest(AdvManifest):
    def __init__(self, max_range: float = 50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_range = max_range

    def select(self, objects: "DataContainer", *args, **kwargs) -> List[TargetObject]:
        # sample the number of false negatives
        n_fn_poisson = (
            np.round(self.rng.poisson(self.n_select_poisson))
            if self.n_select_poisson is not None
            else 0
        )
        objs_within_range = [
            obj for obj in objects if obj.position.norm() < self.max_range
        ]
        n_objects = len(objs_within_range)
        n_fn = min(n_objects, max(self.min_select, min(self.max_select, n_fn_poisson)))
        idx_targets = self.rng.choice(
            list(range(len(objs_within_range))), size=n_fn, replace=False
        )
        targets = [
            TargetObject(obj_state=objs_within_range[idx]) for idx in idx_targets
        ]
        return targets


@AVSEC.register_module()
class TranslationManifest(FalseNegativeManifest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
