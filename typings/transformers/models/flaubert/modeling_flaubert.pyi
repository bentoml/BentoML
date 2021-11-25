

from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_outputs import BaseModelOutput
from ..xlm.modeling_xlm import (
    XLMForMultipleChoice,
    XLMForQuestionAnswering,
    XLMForQuestionAnsweringSimple,
    XLMForSequenceClassification,
    XLMForTokenClassification,
    XLMModel,
    XLMWithLMHeadModel,
)
from .configuration_flaubert import FlaubertConfig

""" PyTorch Flaubert model, based on XLM. """
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
FLAUBERT_START_DOCSTRING = ...
FLAUBERT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare Flaubert Model transformer outputting raw hidden-states without any specific head on top.", FLAUBERT_START_DOCSTRING)
class FlaubertModel(XLMModel):
    config_class = FlaubertConfig
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., langs=..., token_type_ids=..., position_ids=..., lengths=..., cache=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


@add_start_docstrings("""
    The Flaubert Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """, FLAUBERT_START_DOCSTRING)
class FlaubertWithLMHeadModel(XLMWithLMHeadModel):
    """
    This class overrides :class:`~transformers.XLMWithLMHeadModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """
    config_class = FlaubertConfig
    def __init__(self, config) -> None:
        ...
    


@add_start_docstrings("""
    Flaubert Model with a sequence classification/regression head on top (a linear layer on top of the pooled output)
    e.g. for GLUE tasks.
    """, FLAUBERT_START_DOCSTRING)
class FlaubertForSequenceClassification(XLMForSequenceClassification):
    """
    This class overrides :class:`~transformers.XLMForSequenceClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = FlaubertConfig
    def __init__(self, config) -> None:
        ...
    


@add_start_docstrings("""
    Flaubert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, FLAUBERT_START_DOCSTRING)
class FlaubertForTokenClassification(XLMForTokenClassification):
    """
    This class overrides :class:`~transformers.XLMForTokenClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = FlaubertConfig
    def __init__(self, config) -> None:
        ...
    


@add_start_docstrings("""
    Flaubert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, FLAUBERT_START_DOCSTRING)
class FlaubertForQuestionAnsweringSimple(XLMForQuestionAnsweringSimple):
    """
    This class overrides :class:`~transformers.XLMForQuestionAnsweringSimple`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = FlaubertConfig
    def __init__(self, config) -> None:
        ...
    


@add_start_docstrings("""
    Flaubert Model with a beam-search span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, FLAUBERT_START_DOCSTRING)
class FlaubertForQuestionAnswering(XLMForQuestionAnswering):
    """
    This class overrides :class:`~transformers.XLMForQuestionAnswering`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = FlaubertConfig
    def __init__(self, config) -> None:
        ...
    


@add_start_docstrings("""
    Flaubert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, FLAUBERT_START_DOCSTRING)
class FlaubertForMultipleChoice(XLMForMultipleChoice):
    """
    This class overrides :class:`~transformers.XLMForMultipleChoice`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """
    config_class = FlaubertConfig
    def __init__(self, config) -> None:
        ...
    


