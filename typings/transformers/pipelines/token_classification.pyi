

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

if TYPE_CHECKING:
    ...
if is_tf_available():
    ...
if is_torch_available():
    ...
class TokenClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for token classification.
    """
    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        ...
    


class AggregationStrategy(ExplicitEnum):
    """All the valid aggregation strategies for TokenClassificationPipeline"""
    NONE = ...
    SIMPLE = ...
    FIRST = ...
    AVERAGE = ...
    MAX = ...


@add_end_docstrings(PIPELINE_INIT_ARGS, r"""
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
    """)
class TokenClassificationPipeline(Pipeline):
    """
    Named Entity Recognition pipeline using any :obj:`ModelForTokenClassification`. See the `named entity recognition
    examples <../task_summary.html#named-entity-recognition>`__ for more information.

    This token recognition pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"ner"` (for predicting the classes of tokens in a sequence: person, organisation, location
    or miscellaneous).

    The models that this pipeline can use are models that have been fine-tuned on a token classification task. See the
    up-to-date list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=token-classification>`__.
    """
    default_input_names = ...
    def __init__(self, model: Union[PreTrainedModel, TFPreTrainedModel], tokenizer: PreTrainedTokenizer, modelcard: Optional[ModelCard] = ..., framework: Optional[str] = ..., args_parser: ArgumentHandler = ..., device: int = ..., binary_output: bool = ..., ignore_labels=..., task: str = ..., grouped_entities: Optional[bool] = ..., ignore_subwords: Optional[bool] = ..., aggregation_strategy: Optional[AggregationStrategy] = ...) -> None:
        ...
    
    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        """
        Classify each token of the text(s) given as inputs.

        Args:
            inputs (:obj:`str` or :obj:`List[str]`):
                One or several texts (or one list of texts) for token classification.

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a list of dictionaries (one for each token in
            the corresponding input, or each entity if this pipeline was instantiated with an aggregation_strategy)
            with the following keys:

            - **word** (:obj:`str`) -- The token/word classified.
            - **score** (:obj:`float`) -- The corresponding probability for :obj:`entity`.
            - **entity** (:obj:`str`) -- The entity predicted for that token/word (it is named `entity_group` when
              `aggregation_strategy` is not :obj:`"none"`.
            - **index** (:obj:`int`, only present when ``aggregation_strategy="none"``) -- The index of the
              corresponding token in the sentence.
            - **start** (:obj:`int`, `optional`) -- The index of the start of the corresponding entity in the sentence.
              Only exists if the offsets are available within the tokenizer
            - **end** (:obj:`int`, `optional`) -- The index of the end of the corresponding entity in the sentence.
              Only exists if the offsets are available within the tokenizer
        """
        ...
    
    def gather_pre_entities(self, sentence: str, input_ids: np.ndarray, scores: np.ndarray, offset_mapping: Optional[List[Tuple[int, int]]], special_tokens_mask: np.ndarray) -> List[dict]:
        """Fuse various numpy arrays into dicts with all the information needed for aggregation"""
        ...
    
    def aggregate(self, pre_entities: List[dict], aggregation_strategy: AggregationStrategy) -> List[dict]:
        ...
    
    def aggregate_word(self, entities: List[dict], aggregation_strategy: AggregationStrategy) -> dict:
        ...
    
    def aggregate_words(self, entities: List[dict], aggregation_strategy: AggregationStrategy) -> List[dict]:
        """
        Override tokens from a given word that disagree to force agreement on word boundaries.

        Example: micro|soft| com|pany| B-ENT I-NAME I-ENT I-ENT will be rewritten with first strategy as microsoft|
        company| B-ENT I-ENT
        """
        ...
    
    def group_sub_entities(self, entities: List[dict]) -> dict:
        """
        Group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """
        ...
    
    def get_tag(self, entity_name: str) -> Tuple[str, str]:
        ...
    
    def group_entities(self, entities: List[dict]) -> List[dict]:
        """
        Find and group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """
        ...
    


NerPipeline = TokenClassificationPipeline
