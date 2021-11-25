

from typing import Mapping

from ...onnx import OnnxConfig
from ..roberta.configuration_roberta import RobertaConfig

logger = ...
XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class XLMRobertaConfig(RobertaConfig):
    """
    This class overrides :class:`~transformers.RobertaConfig`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """
    model_type = ...


class XLMRobertaOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        ...
    
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        ...
    


