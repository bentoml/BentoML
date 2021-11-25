

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
from .configuration_camembert import CamembertConfig

"""PyTorch CamemBERT model. """
logger = ...
_TOKENIZER_FOR_DOC = ...
CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
CAMEMBERT_START_DOCSTRING = ...
@add_start_docstrings("The bare CamemBERT Model transformer outputting raw hidden-states without any specific head on top.", CAMEMBERT_START_DOCSTRING)
class CamembertModel(RobertaModel):
    """
    This class overrides :class:`~transformers.RobertaModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """
    config_class = CamembertConfig


@add_start_docstrings("""CamemBERT Model with a `language modeling` head on top. """, CAMEMBERT_START_DOCSTRING)
class CamembertForMaskedLM(RobertaForMaskedLM):
    """
    This class overrides :class:`~transformers.RobertaForMaskedLM`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """
    config_class = CamembertConfig


@add_start_docstrings("""
    CamemBERT Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """, CAMEMBERT_START_DOCSTRING)
class CamembertForSequenceClassification(RobertaForSequenceClassification):
    """
    This class overrides :class:`~transformers.RobertaForSequenceClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = CamembertConfig


@add_start_docstrings("""
    CamemBERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, CAMEMBERT_START_DOCSTRING)
class CamembertForMultipleChoice(RobertaForMultipleChoice):
    """
    This class overrides :class:`~transformers.RobertaForMultipleChoice`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = CamembertConfig


@add_start_docstrings("""
    CamemBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """, CAMEMBERT_START_DOCSTRING)
class CamembertForTokenClassification(RobertaForTokenClassification):
    """
    This class overrides :class:`~transformers.RobertaForTokenClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = CamembertConfig


@add_start_docstrings("""
    CamemBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`
    """, CAMEMBERT_START_DOCSTRING)
class CamembertForQuestionAnswering(RobertaForQuestionAnswering):
    """
    This class overrides :class:`~transformers.RobertaForQuestionAnswering`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = CamembertConfig


@add_start_docstrings("""CamemBERT Model with a `language modeling` head on top for CLM fine-tuning. """, CAMEMBERT_START_DOCSTRING)
class CamembertForCausalLM(RobertaForCausalLM):
    """
    This class overrides :class:`~transformers.RobertaForCausalLM`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """
    config_class = CamembertConfig


