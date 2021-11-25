

from ...file_utils import add_start_docstrings
from ..roberta.modeling_tf_roberta import (
    TFRobertaForMaskedLM,
    TFRobertaForMultipleChoice,
    TFRobertaForQuestionAnswering,
    TFRobertaForSequenceClassification,
    TFRobertaForTokenClassification,
    TFRobertaModel,
)
from .configuration_xlm_roberta import XLMRobertaConfig

""" TF 2.0  XLM-RoBERTa model. """
logger = ...
TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = ...
XLM_ROBERTA_START_DOCSTRING = ...
@add_start_docstrings("The bare XLM-RoBERTa Model transformer outputting raw hidden-states without any specific head on top.", XLM_ROBERTA_START_DOCSTRING)
class TFXLMRobertaModel(TFRobertaModel):
    """
    This class overrides :class:`~transformers.TFRobertaModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig


@add_start_docstrings("""XLM-RoBERTa Model with a `language modeling` head on top. """, XLM_ROBERTA_START_DOCSTRING)
class TFXLMRobertaForMaskedLM(TFRobertaForMaskedLM):
    """
    This class overrides :class:`~transformers.TFRobertaForMaskedLM`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig


@add_start_docstrings("""
    XLM-RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """, XLM_ROBERTA_START_DOCSTRING)
class TFXLMRobertaForSequenceClassification(TFRobertaForSequenceClassification):
    """
    This class overrides :class:`~transformers.TFRobertaForSequenceClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig


@add_start_docstrings("""
    XLM-RoBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """, XLM_ROBERTA_START_DOCSTRING)
class TFXLMRobertaForTokenClassification(TFRobertaForTokenClassification):
    """
    This class overrides :class:`~transformers.TFRobertaForTokenClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig


@add_start_docstrings("""
XLM-RoBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
""", XLM_ROBERTA_START_DOCSTRING)
class TFXLMRobertaForQuestionAnswering(TFRobertaForQuestionAnswering):
    """
    This class overrides :class:`~transformers.TFRobertaForQuestionAnsweringSimple`. Please check the superclass for
    the appropriate documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig


@add_start_docstrings("""
    Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, XLM_ROBERTA_START_DOCSTRING)
class TFXLMRobertaForMultipleChoice(TFRobertaForMultipleChoice):
    """
    This class overrides :class:`~transformers.TFRobertaForMultipleChoice`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig


