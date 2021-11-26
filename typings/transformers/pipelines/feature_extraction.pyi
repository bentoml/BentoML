from typing import TYPE_CHECKING, Optional, Union
from ..modelcard import ModelCard
from ..modeling_tf_utils import TFPreTrainedModel
from ..modeling_utils import PreTrainedModel
from ..tokenization_utils import PreTrainedTokenizer
from .base import ArgumentHandler, Pipeline

if TYPE_CHECKING: ...

class FeatureExtractionPipeline(Pipeline):
    def __init__(
        self,
        model: Union[PreTrainedModel, TFPreTrainedModel],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = ...,
        framework: Optional[str] = ...,
        args_parser: ArgumentHandler = ...,
        device: int = ...,
        task: str = ...,
    ) -> None: ...
