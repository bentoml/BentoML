

from ..prophetnet.modeling_prophetnet import (
    ProphetNetDecoder,
    ProphetNetEncoder,
    ProphetNetForCausalLM,
    ProphetNetForConditionalGeneration,
    ProphetNetModel,
)
from .configuration_xlm_prophetnet import XLMProphetNetConfig

logger = ...
_TOKENIZER_FOR_DOC = ...
XLM_PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class XLMProphetNetEncoder(ProphetNetEncoder):
    r"""
    This class overrides :class:`~transformers.ProphetNetEncoder`. Please check the superclass for the appropriate
    documentation alongside usage examples.

    Example::

        >>> from transformers import XLMProphetNetTokenizer, XLMProphetNetEncoder
        >>> import torch

        >>> tokenizer = XLMProphetNetTokenizer.from_pretrained('microsoft/xprophetnet-large-wiki100-cased')
        >>> model = XLMProphetNetEncoder.from_pretrained('patrickvonplaten/xprophetnet-large-uncased-standalone')
        >>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
    """
    config_class = XLMProphetNetConfig


class XLMProphetNetDecoder(ProphetNetDecoder):
    r"""
    This class overrides :class:`~transformers.ProphetNetDecoder`. Please check the superclass for the appropriate
    documentation alongside usage examples.

    Example::

        >>> from transformers import XLMProphetNetTokenizer, XLMProphetNetDecoder
        >>> import torch

        >>> tokenizer = XLMProphetNetTokenizer.from_pretrained('microsoft/xprophetnet-large-wiki100-cased')
        >>> model = XLMProphetNetDecoder.from_pretrained('patrickvonplaten/xprophetnet-large-uncased-standalone', add_cross_attention=False)
        >>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
    """
    config_class = XLMProphetNetConfig


class XLMProphetNetModel(ProphetNetModel):
    r"""
    This class overrides :class:`~transformers.ProphetNetModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.

    Example::

        >>> from transformers import XLMProphetNetTokenizer, XLMProphetNetModel

        >>> tokenizer = XLMProphetNetTokenizer.from_pretrained('microsoft/xprophetnet-large-wiki100-cased')
        >>> model = XLMProphetNetModel.from_pretrained('microsoft/xprophetnet-large-wiki100-cased')

        >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

        >>> last_hidden_states = outputs.last_hidden_state  # main stream hidden states
            >>> last_hidden_states_ngram = outputs.last_hidden_state_ngram  # predict hidden states
    """
    config_class = XLMProphetNetConfig


class XLMProphetNetForConditionalGeneration(ProphetNetForConditionalGeneration):
    r"""
    This class overrides :class:`~transformers.ProphetNetForConditionalGeneration`. Please check the superclass for the
    appropriate documentation alongside usage examples.

    Example::

        >>> from transformers import XLMProphetNetTokenizer, XLMProphetNetForConditionalGeneration

        >>> tokenizer = XLMProphetNetTokenizer.from_pretrained('microsoft/xprophetnet-large-wiki100-cased')
        >>> model =  XLMProphetNetForConditionalGeneration.from_pretrained('microsoft/xprophetnet-large-wiki100-cased')

        >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

        >>> logits_next_token = outputs.logits  # logits to predict next token as usual
        >>> logits_ngram_next_tokens = outputs.logits_ngram  # logits to predict 2nd, 3rd, ... next tokens
    """
    config_class = XLMProphetNetConfig


class XLMProphetNetForCausalLM(ProphetNetForCausalLM):
    r"""
    This class overrides :class:`~transformers.ProphetNetForCausalLM`. Please check the superclass for the appropriate
    documentation alongside usage examples.

    Example::

        >>> from transformers import XLMProphetNetTokenizer, XLMProphetNetForCausalLM
        >>> import torch

        >>> tokenizer = XLMProphetNetTokenizer.from_pretrained('microsoft/xprophetnet-large-wiki100-cased')
        >>> model = XLMProphetNetForCausalLM.from_pretrained('microsoft/xprophetnet-large-wiki100-cased')
        >>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> logits = outputs.logits

        >>> # Model can also be used with EncoderDecoder framework
        >>> from transformers import EncoderDecoderModel, XLMProphetNetTokenizer, XLMRobertaTokenizer
        >>> import torch

        >>> tokenizer_enc = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        >>> tokenizer_dec = XLMProphetNetTokenizer.from_pretrained('microsoft/xprophetnet-large-wiki100-cased')
        >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("xlm-roberta-large", 'microsoft/xprophetnet-large-wiki100-cased')

        >>> ARTICLE = (
        ... "the us state department said wednesday it had received no "
        ... "formal word from bolivia that it was expelling the us ambassador there "
        ... "but said the charges made against him are `` baseless ."
        ... )
        >>> input_ids = tokenizer_enc(ARTICLE, return_tensors="pt").input_ids
        >>> labels = tokenizer_dec("us rejects charges against its ambassador in bolivia", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=labels[:, :-1], labels=labels[:, 1:])

        >>> loss = outputs.loss
    """
    config_class = XLMProphetNetConfig


