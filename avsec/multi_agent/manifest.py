from typing import TYPE_CHECKING, List


if TYPE_CHECKING:
    from avstack.datastructs import DataContainer
    from avstack.geometry import ReferenceFrame

import numpy as np
from avstack.environment.objects import ObjectState
from avstack.geometry import Attitude, Box3D, Position

from avsec.config import AVSEC

from .target import TargetObject


class AdvManifest:
    def select(self):
        raise NotImplementedError


@AVSEC.register_module()
class FalsePositiveManifest(AdvManifest):
    def __init__(
        self, fp_poisson: float, x_sigma: float = 30, hwl: List[float] = [2, 2, 4]
    ):
        self.fp_poisson = fp_poisson
        self.x_sigma = x_sigma
        self.hwl = hwl

    def select(
        self,
        timestamp: float,
        reference_agent: "ReferenceFrame",
        min_select: int = 0,
        *args,
        **kwargs,
    ) -> List[TargetObject]:
        """Create some false positives for attacks.

        Keep in mind the reference frame may not be parallel to the group plane.
        Therefore, the generation of false positive targets should be assumed
        to be made relative to a projected sensor plane that is co-planar with ground.
        """
        reference_agent_gp = reference_agent.get_ground_projected_reference()

        # sample the number of false positives
        n_fp = max(min_select, int(np.random.poisson(self.fp_poisson)))

        # construct target positions
        targets = []
        for i in range(n_fp):
            # get data
            x_vec = self.x_sigma * np.array([np.random.randn(), np.random.randn(), 0])
            q_vec = np.quaternion(1)  # identity until velocity set later

            # adjust for false positive selection that is coplanar with ground
            pos = Position(x_vec, reference=reference_agent_gp).change_reference(
                reference_agent, inplace=False
            )
            att = Attitude(q_vec, reference=reference_agent_gp).change_reference(
                reference_agent, inplace=False
            )

            # populate object state
            obj = ObjectState(ID=i, obj_type="car")
            obj.set(
                t=timestamp, position=pos, box=Box3D(pos, att, self.hwl), attitude=att
            )
            targets.append(TargetObject(obj_state=obj))
        return targets


@AVSEC.register_module()
class FalseNegativeManifest(AdvManifest):
    def __init__(self, fn_fraction: float):
        self.fn_fraction = fn_fraction

    def select(
        self, objects: "DataContainer", min_select: int = 0, *args, **kwargs
    ) -> List[TargetObject]:
        n_objects = len(objects)
        n_fn = max(min_select, int(self.fn_fraction * n_objects))
        idx_targets = np.random.choice(
            list(range(len(objects))), size=n_fn, replace=False
        )
        targets = [TargetObject(obj_state=objects[idx]) for idx in idx_targets]
        return targets


@AVSEC.register_module()
class TranslationManifest(FalseNegativeManifest):
    def __init__(self, tr_fraction: float):
        super().__init__(fn_fraction=tr_fraction)
