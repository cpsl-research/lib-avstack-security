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
        min_select: int = 0,
        max_select: int = np.inf,
    ):
        self.rng = np.random.RandomState(seed)
        self.min_select = min_select
        self.max_select = max_select

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
        fp_poisson: float,
        x_sigma: float = 30,
        hwl: List[float] = [2, 2, 4],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fp_poisson = fp_poisson
        self.x_sigma = x_sigma
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
        n_fp = max(self.min_select, np.round(self.rng.poisson(self.fp_poisson)))

        # construct target positions
        targets = []
        for i in range(n_fp):
            # get data -- flat on ground plane
            x_vec = self.x_sigma * np.array([self.rng.randn(), self.rng.randn(), 0])
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
    def __init__(self, fn_poisson: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn_poisson = fn_poisson

    def select(self, objects: "DataContainer", *args, **kwargs) -> List[TargetObject]:
        n_objects = len(objects)
        n_fn = max(
            self.min_select,
            min(min(n_objects, self.max_select), self.rng.poisson(self.fn_poisson)),
        )
        idx_targets = self.rng.choice(
            list(range(len(objects))), size=n_fn, replace=False
        )
        targets = [TargetObject(obj_state=objects[idx]) for idx in idx_targets]
        return targets


@AVSEC.register_module()
class TranslationManifest(FalseNegativeManifest):
    def __init__(self, tr_poisson: float, *args, **kwargs):
        super().__init__(fn_poisson=tr_poisson, *args, **kwargs)
