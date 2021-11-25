

from ..t5.modeling_tf_t5 import (
    TFT5EncoderModel,
    TFT5ForConditionalGeneration,
    TFT5Model,
)
from .configuration_mt5 import MT5Config

""" Tensorflow mT5 model. """
logger = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
class TFMT5Model(TFT5Model):
    r"""
    This class overrides :class:`~transformers.TFT5Model`. Please check the superclass for the appropriate
    documentation alongside usage examples.

    Examples::

        >>> from transformers import TFMT5Model, T5Tokenizer
        >>> model = TFMT5Model.from_pretrained("google/mt5-small")
        >>> tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
        >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
        >>> summary = "Weiter Verhandlung in Syrien."
        >>> inputs = tokenizer(article, return_tensors="tf")
        >>> with tokenizer.as_target_tokenizer():
        ...     labels = tokenizer(summary, return_tensors="tf")

        >>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
        >>> hidden_states = outputs.last_hidden_state
    """
    model_type = ...
    config_class = MT5Config


class TFMT5ForConditionalGeneration(TFT5ForConditionalGeneration):
    r"""
    This class overrides :class:`~transformers.TFT5ForConditionalGeneration`. Please check the superclass for the
    appropriate documentation alongside usage examples.

    Examples::

        >>> from transformers import TFMT5ForConditionalGeneration, T5Tokenizer
        >>> model = TFMT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        >>> tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
        >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
        >>> summary = "Weiter Verhandlung in Syrien."
        >>> inputs = tokenizer(article, return_tensors="tf")
        >>> with tokenizer.as_target_tokenizer():
        ...     labels = tokenizer(summary, return_tensors="tf")

        >>> outputs = model(**inputs,labels=labels["input_ids"])
        >>> loss = outputs.loss
    """
    model_type = ...
    config_class = MT5Config


class TFMT5EncoderModel(TFT5EncoderModel):
    r"""
    This class overrides :class:`~transformers.TFT5EncoderModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.

    Examples::

        >>> from transformers import TFMT5EncoderModel, T5Tokenizer
        >>> model = TFMT5EncoderModel.from_pretrained("google/mt5-small")
        >>> tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
        >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
        >>> input_ids = tokenizer(article, return_tensors="tf").input_ids
        >>> outputs = model(input_ids)
        >>> hidden_state = outputs.last_hidden_state
    """
    model_type = ...
    config_class = MT5Config


