from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import numpy as np
from ..data import SquadExample
from ..file_utils import add_end_docstrings, is_tf_available, is_torch_available
from ..modelcard import ModelCard
from ..modeling_tf_utils import TFPreTrainedModel
from ..modeling_utils import PreTrainedModel
from ..tokenization_utils import PreTrainedTokenizer
from .base import PIPELINE_INIT_ARGS, ArgumentHandler, Pipeline

if TYPE_CHECKING: ...
if is_tf_available(): ...
if is_torch_available(): ...

class QuestionAnsweringArgumentHandler(ArgumentHandler):
    def normalize(self, item): ...
    def __call__(self, *args, **kwargs): ...

@add_end_docstrings(PIPELINE_INIT_ARGS)
class QuestionAnsweringPipeline(Pipeline):
    default_input_names = ...
    def __init__(
        self,
        model: Union[PreTrainedModel, TFPreTrainedModel],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = ...,
        framework: Optional[str] = ...,
        device: int = ...,
        task: str = ...,
        **kwargs
    ) -> None: ...
    @staticmethod
    def create_sample(
        question: Union[str, List[str]], context: Union[str, List[str]]
    ) -> Union[SquadExample, List[SquadExample]]: ...
    def __call__(self, *args, **kwargs): ...
    def decode(
        self,
        start: np.ndarray,
        end: np.ndarray,
        topk: int,
        max_answer_len: int,
        undesired_tokens: np.ndarray,
    ) -> Tuple: ...
    def span_to_answer(
        self, text: str, start: int, end: int
    ) -> Dict[str, Union[str, int]]: ...
