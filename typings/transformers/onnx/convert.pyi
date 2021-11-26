from pathlib import Path
from typing import Iterable, List, Tuple, Union
from packaging.version import Version
from .. import PreTrainedModel, PreTrainedTokenizer, TFPreTrainedModel
from .config import OnnxConfig

logger = ...
ORT_QUANTIZE_MINIMUM_VERSION = ...

def check_onnxruntime_requirements(minimum_version: Version): ...
def export(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    config: OnnxConfig,
    opset: int,
    output: Path,
) -> Tuple[List[str], List[str]]: ...
def validate_model_outputs(
    config: OnnxConfig,
    tokenizer: PreTrainedTokenizer,
    reference_model: Union[PreTrainedModel, TFPreTrainedModel],
    onnx_model: Path,
    onnx_named_outputs: List[str],
    atol: float,
): ...
def ensure_model_and_config_inputs_match(
    model: Union[PreTrainedModel, TFPreTrainedModel], model_inputs: Iterable[str]
) -> Tuple[bool, List[str]]: ...
