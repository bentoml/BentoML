

from ...file_utils import add_start_docstrings
from ..roberta.modeling_tf_roberta import (
    TFRobertaForMaskedLM,
    TFRobertaForMultipleChoice,
    TFRobertaForQuestionAnswering,
    TFRobertaForSequenceClassification,
    TFRobertaForTokenClassification,
    TFRobertaModel,
)
from .configuration_camembert import CamembertConfig

""" TF 2.0 CamemBERT model. """
logger = ...
TF_CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
CAMEMBERT_START_DOCSTRING = ...
@add_start_docstrings("The bare CamemBERT Model transformer outputting raw hidden-states without any specific head on top.", CAMEMBERT_START_DOCSTRING)
class TFCamembertModel(TFRobertaModel):
    """
    This class overrides :class:`~transformers.TFRobertaModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """
    config_class = CamembertConfig


@add_start_docstrings("""CamemBERT Model with a `language modeling` head on top. """, CAMEMBERT_START_DOCSTRING)
class TFCamembertForMaskedLM(TFRobertaForMaskedLM):
    """
    This class overrides :class:`~transformers.TFRobertaForMaskedLM`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """
    config_class = CamembertConfig


@add_start_docstrings("""
    CamemBERT Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """, CAMEMBERT_START_DOCSTRING)
class TFCamembertForSequenceClassification(TFRobertaForSequenceClassification):
    """
    This class overrides :class:`~transformers.TFRobertaForSequenceClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = CamembertConfig


@add_start_docstrings("""
    CamemBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """, CAMEMBERT_START_DOCSTRING)
class TFCamembertForTokenClassification(TFRobertaForTokenClassification):
    """
    This class overrides :class:`~transformers.TFRobertaForTokenClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = CamembertConfig


@add_start_docstrings("""
    CamemBERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, CAMEMBERT_START_DOCSTRING)
class TFCamembertForMultipleChoice(TFRobertaForMultipleChoice):
    """
    This class overrides :class:`~transformers.TFRobertaForMultipleChoice`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = CamembertConfig


@add_start_docstrings("""
    CamemBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, CAMEMBERT_START_DOCSTRING)
class TFCamembertForQuestionAnswering(TFRobertaForQuestionAnswering):
    """
    This class overrides :class:`~transformers.TFRobertaForQuestionAnswering`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = CamembertConfig


