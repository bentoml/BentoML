from typing import TYPE_CHECKING, Optional, Union
import numpy as np
from ..file_utils import add_end_docstrings, is_tf_available, is_torch_available
from ..modelcard import ModelCard
from ..modeling_tf_utils import TFPreTrainedModel
from ..modeling_utils import PreTrainedModel
from ..tokenization_utils import PreTrainedTokenizer
from .base import PIPELINE_INIT_ARGS, ArgumentHandler, Pipeline

if TYPE_CHECKING: ...
if is_tf_available(): ...
if is_torch_available(): ...
logger = ...

@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        top_k (:obj:`int`, defaults to 5): The number of predictions to return.
    """,
)
class FillMaskPipeline(Pipeline):
    def __init__(
        self,
        model: Union[PreTrainedModel, TFPreTrainedModel],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = ...,
        framework: Optional[str] = ...,
        args_parser: ArgumentHandler = ...,
        device: int = ...,
        top_k=...,
        task: str = ...,
    ) -> None: ...
    def ensure_exactly_one_mask_token(self, masked_index: np.ndarray): ...
    def __call__(self, *args, targets=..., top_k: Optional[int] = ..., **kwargs): ...
