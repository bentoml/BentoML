

from ...file_utils import add_start_docstrings
from ..roberta.modeling_roberta import (
    RobertaForCausalLM,
    RobertaForMaskedLM,
    RobertaForMultipleChoice,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaModel,
)
from .configuration_xlm_roberta import XLMRobertaConfig

"""PyTorch XLM-RoBERTa model. """
logger = ...
XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = ...
XLM_ROBERTA_START_DOCSTRING = ...
@add_start_docstrings("The bare XLM-RoBERTa Model transformer outputting raw hidden-states without any specific head on top.", XLM_ROBERTA_START_DOCSTRING)
class XLMRobertaModel(RobertaModel):
    """
    This class overrides :class:`~transformers.RobertaModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig


@add_start_docstrings("XLM-RoBERTa Model with a `language modeling` head on top for CLM fine-tuning.", XLM_ROBERTA_START_DOCSTRING)
class XLMRobertaForCausalLM(RobertaForCausalLM):
    """
    This class overrides :class:`~transformers.RobertaForCausalLM`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig


@add_start_docstrings("""XLM-RoBERTa Model with a `language modeling` head on top. """, XLM_ROBERTA_START_DOCSTRING)
class XLMRobertaForMaskedLM(RobertaForMaskedLM):
    """
    This class overrides :class:`~transformers.RobertaForMaskedLM`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig


@add_start_docstrings("""
    XLM-RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """, XLM_ROBERTA_START_DOCSTRING)
class XLMRobertaForSequenceClassification(RobertaForSequenceClassification):
    """
    This class overrides :class:`~transformers.RobertaForSequenceClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig


@add_start_docstrings("""
    XLM-RoBERTa Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """, XLM_ROBERTA_START_DOCSTRING)
class XLMRobertaForMultipleChoice(RobertaForMultipleChoice):
    """
    This class overrides :class:`~transformers.RobertaForMultipleChoice`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig


@add_start_docstrings("""
    XLM-RoBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """, XLM_ROBERTA_START_DOCSTRING)
class XLMRobertaForTokenClassification(RobertaForTokenClassification):
    """
    This class overrides :class:`~transformers.RobertaForTokenClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig


@add_start_docstrings("""
    XLM-RoBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, XLM_ROBERTA_START_DOCSTRING)
class XLMRobertaForQuestionAnswering(RobertaForQuestionAnswering):
    """
    This class overrides :class:`~transformers.RobertaForQuestionAnswering`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig


