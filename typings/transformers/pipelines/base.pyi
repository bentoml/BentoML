from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union
from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..file_utils import add_end_docstrings
from ..modelcard import ModelCard
from ..modeling_tf_utils import TFPreTrainedModel
from ..modeling_utils import PreTrainedModel
from ..models.auto.configuration_auto import AutoConfig
from ..tokenization_utils import PreTrainedTokenizer

def infer_framework_load_model(
    model,
    config: AutoConfig,
    model_classes: Optional[Dict[str, Tuple[type]]] = ...,
    task: Optional[str] = ...,
    framework: Optional[str] = ...,
    **model_kwargs
): ...
def infer_framework_from_model(
    model,
    model_classes: Optional[Dict[str, Tuple[type]]] = ...,
    task: Optional[str] = ...,
    framework: Optional[str] = ...,
    **model_kwargs
): ...
def get_framework(model, revision: Optional[str] = ...): ...
def get_default_model(
    targeted_task: Dict, framework: Optional[str], task_options: Optional[Any]
) -> str: ...

class PipelineException(Exception):
    def __init__(self, task: str, model: str, reason: str) -> None: ...

class ArgumentHandler(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs): ...

class PipelineDataFormat:
    SUPPORTED_FORMATS = ...
    def __init__(
        self,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite: bool = ...,
    ) -> None: ...
    @abstractmethod
    def __iter__(self): ...
    @abstractmethod
    def save(self, data: Union[dict, List[dict]]): ...
    def save_binary(self, data: Union[dict, List[dict]]) -> str: ...
    @staticmethod
    def from_str(
        format: str,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite=...,
    ) -> PipelineDataFormat: ...

class CsvPipelineDataFormat(PipelineDataFormat):
    def __init__(
        self,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite=...,
    ) -> None: ...
    def __iter__(self): ...
    def save(self, data: List[dict]): ...

class JsonPipelineDataFormat(PipelineDataFormat):
    def __init__(
        self,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite=...,
    ) -> None: ...
    def __iter__(self): ...
    def save(self, data: dict): ...

class PipedPipelineDataFormat(PipelineDataFormat):
    def __iter__(self): ...
    def save(self, data: dict): ...
    def save_binary(self, data: Union[dict, List[dict]]) -> str: ...

class _ScikitCompat(ABC):
    @abstractmethod
    def transform(self, X): ...
    @abstractmethod
    def predict(self, X): ...

PIPELINE_INIT_ARGS: str = ...

class Pipeline(_ScikitCompat):
    default_input_names: None = ...
    model: Union[PreTrainedModel, TFPreTrainedModel]
    tokenizer: Optional[PreTrainedTokenizer]
    feature_extractor: Optional[PreTrainedFeatureExtractor]
    def __init__(
        self,
        model: Union[PreTrainedModel, TFPreTrainedModel],
        tokenizer: Optional[PreTrainedTokenizer] = ...,
        feature_extractor: Optional[PreTrainedFeatureExtractor] = ...,
        modelcard: Optional[ModelCard] = ...,
        framework: Optional[str] = ...,
        task: str = ...,
        args_parser: ArgumentHandler = ...,
        device: int = ...,
        binary_output: bool = ...,
    ) -> None: ...
    def save_pretrained(self, save_directory: str) -> None: ...
    def transform(self, X: Any) -> Any: ...
    def predict(self, X: Any) -> Any: ...
    @contextmanager
    def device_placement(self) -> None: ...
    def ensure_tensor_on_device(self, **inputs: Any) -> None: ...
    def check_model_type(
        self, supported_models: Union[List[str], Dict[str, Any]]
    ) -> bool: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
