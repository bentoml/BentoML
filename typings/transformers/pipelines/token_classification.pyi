from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import numpy as np
from ..file_utils import (
    ExplicitEnum,
    add_end_docstrings,
    is_tf_available,
    is_torch_available,
)
from ..modelcard import ModelCard
from ..modeling_tf_utils import TFPreTrainedModel
from ..modeling_utils import PreTrainedModel
from ..tokenization_utils import PreTrainedTokenizer
from .base import PIPELINE_INIT_ARGS, ArgumentHandler, Pipeline

if TYPE_CHECKING: ...
if is_tf_available(): ...
if is_torch_available(): ...

class TokenClassificationArgumentHandler(ArgumentHandler):
    def __call__(self, inputs: Union[str, List[str]], **kwargs): ...

class AggregationStrategy(ExplicitEnum):
    NONE = ...
    SIMPLE = ...
    FIRST = ...
    AVERAGE = ...
    MAX = ...

@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        ignore_labels (:obj:`List[str]`, defaults to :obj:`["O"]`):
            A list of labels to ignore.
        grouped_entities (:obj:`bool`, `optional`, defaults to :obj:`False`):
            DEPRECATED, use :obj:`aggregation_strategy` instead. Whether or not to group the tokens corresponding to
            the same entity together in the predictions or not.
        aggregation_strategy (:obj:`str`, `optional`, defaults to :obj:`"none"`): The strategy to fuse (or not) tokens based on the model prediction.
                - "none" : Will simply not do any aggregation and simply return raw results from the model
                - "simple" : Will attempt to group entities following the default schema. (A, B-TAG), (B, I-TAG), (C,
                  I-TAG), (D, B-TAG2) (E, B-TAG2) will end up being [{"word": ABC, "entity": "TAG"}, {"word": "D",
                  "entity": "TAG2"}, {"word": "E", "entity": "TAG2"}] Notice that two consecutive B tags will end up as
                  different entities. On word based languages, we might end up splitting words undesirably : Imagine
                  Microsoft being tagged as [{"word": "Micro", "entity": "ENTERPRISE"}, {"word": "soft", "entity":
                  "NAME"}]. Look for FIRST, MAX, AVERAGE for ways to mitigate that and disambiguate words (on languages
                  that support that meaning, which is basically tokens separated by a space). These mitigations will
                  only work on real words, "New york" might still be tagged with two different entities.
                - "first" : (works only on word based models) Will use the :obj:`SIMPLE` strategy except that words,
                  cannot end up with different tags. Words will simply use the tag of the first token of the word when
                  there is ambiguity.
                - "average" : (works only on word based models) Will use the :obj:`SIMPLE` strategy except that words,
                  cannot end up with different tags. scores will be averaged first across tokens, and then the maximum
                  label is applied.
                - "max" : (works only on word based models) Will use the :obj:`SIMPLE` strategy except that words,
                  cannot end up with different tags. Word entity will simply be the token with the maximum score.
    """,
)
class TokenClassificationPipeline(Pipeline):
    default_input_names = ...
    def __init__(
        self,
        model: Union[PreTrainedModel, TFPreTrainedModel],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = ...,
        framework: Optional[str] = ...,
        args_parser: ArgumentHandler = ...,
        device: int = ...,
        binary_output: bool = ...,
        ignore_labels=...,
        task: str = ...,
        grouped_entities: Optional[bool] = ...,
        ignore_subwords: Optional[bool] = ...,
        aggregation_strategy: Optional[AggregationStrategy] = ...,
    ) -> None: ...
    def __call__(self, inputs: Union[str, List[str]], **kwargs): ...
    def gather_pre_entities(
        self,
        sentence: str,
        input_ids: np.ndarray,
        scores: np.ndarray,
        offset_mapping: Optional[List[Tuple[int, int]]],
        special_tokens_mask: np.ndarray,
    ) -> List[dict]: ...
    def aggregate(
        self, pre_entities: List[dict], aggregation_strategy: AggregationStrategy
    ) -> List[dict]: ...
    def aggregate_word(
        self, entities: List[dict], aggregation_strategy: AggregationStrategy
    ) -> dict: ...
    def aggregate_words(
        self, entities: List[dict], aggregation_strategy: AggregationStrategy
    ) -> List[dict]: ...
    def group_sub_entities(self, entities: List[dict]) -> dict: ...
    def get_tag(self, entity_name: str) -> Tuple[str, str]: ...
    def group_entities(self, entities: List[dict]) -> List[dict]: ...

NerPipeline = TokenClassificationPipeline
