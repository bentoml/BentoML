

from ..t5.modeling_t5 import T5EncoderModel, T5ForConditionalGeneration, T5Model
from .configuration_mt5 import MT5Config

""" PyTorch mT5 model. """
logger = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
class MT5Model(T5Model):
    r"""
    This class overrides :class:`~transformers.T5Model`. Please check the superclass for the appropriate documentation
    alongside usage examples.

    Examples::

        >>> from transformers import MT5Model, T5Tokenizer
        >>> model = MT5Model.from_pretrained("google/mt5-small")
        >>> tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
        >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
        >>> summary = "Weiter Verhandlung in Syrien."
        >>> inputs = tokenizer(article, return_tensors="pt")
        >>> with tokenizer.as_target_tokenizer():
        ...     labels = tokenizer(summary, return_tensors="pt")

        >>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
        >>> hidden_states = outputs.last_hidden_state
    """
    model_type = ...
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = ...
    _keys_to_ignore_on_save = ...


class MT5ForConditionalGeneration(T5ForConditionalGeneration):
    r"""
    This class overrides :class:`~transformers.T5ForConditionalGeneration`. Please check the superclass for the
    appropriate documentation alongside usage examples.

    Examples::

        >>> from transformers import MT5ForConditionalGeneration, T5Tokenizer
        >>> model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        >>> tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
        >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
        >>> summary = "Weiter Verhandlung in Syrien."
        >>> inputs = tokenizer(article, return_tensors="pt")
        >>> with tokenizer.as_target_tokenizer():
        ...     labels = tokenizer(summary, return_tensors="pt")

        >>> outputs = model(**inputs,labels=labels["input_ids"])
        >>> loss = outputs.loss
    """
    model_type = ...
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = ...
    _keys_to_ignore_on_save = ...


class MT5EncoderModel(T5EncoderModel):
    r"""
    This class overrides :class:`~transformers.T5EncoderModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.

    Examples::

        >>> from transformers import MT5EncoderModel, T5Tokenizer
        >>> model = MT5EncoderModel.from_pretrained("google/mt5-small")
        >>> tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
        >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
        >>> input_ids = tokenizer(article, return_tensors="pt").input_ids
        >>> outputs = model(input_ids)
        >>> hidden_state = outputs.last_hidden_state
    """
    model_type = ...
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = ...
    _keys_to_ignore_on_save = ...


