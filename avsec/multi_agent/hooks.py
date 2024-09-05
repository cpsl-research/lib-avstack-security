from typing import TYPE_CHECKING, Dict, Union


if TYPE_CHECKING:
    from avsec.multi_agent.adversary import AdversaryModel
    from avstack.datastructs import DataContainer
    from avstack.geometry import ReferenceFrame, Shape

from avstack.config import HOOKS

from avsec.config import AVSEC


@HOOKS.register_module()
class AdversaryHook:
    def __init__(
        self, models: Dict[tuple, Union[dict, "AdversaryModel"]], verbose: bool = False
    ):
        self.verbose = verbose
        self.models = {
            k: AVSEC.build(model) if isinstance(model, dict) else model
            for k, model in models.items()
        }
        self.verbose = verbose

    def __call__(
        self,
        detections: "DataContainer",
        field_of_view: "Shape",
        reference: "ReferenceFrame",
        agent_name: str,
        sensor_name: str,
        logger=None,
    ):
        """Call the adversary hook on the detection data

        Args:
            detections: datacontainer of perception detections
            reference: reference frame for agent's sensor
            agent_name: agent's name
            sensor_name: sensor's name
        """
        attacked_agents = set()
        if (agent_name, sensor_name) in self.models:
            n_before = len(detections)
            detections, field_of_view, did_attack = self.models[
                (agent_name, sensor_name)
            ](
                objects=detections,
                fov=field_of_view,
                reference=reference,
            )
            if did_attack:
                attacked_agents.add(agent_name)
            n_after = len(detections)
            if self.verbose:
                if logger is not None:
                    logger.info(f"Detections: {n_before} -> {n_after}")
        else:
            if self.verbose:
                if logger is not None:
                    model_keys = list(self.models.keys())
                    logger.info(
                        f"({agent_name}, {sensor_name}) not in models ({model_keys})"
                    )

        return detections, field_of_view, attacked_agents
