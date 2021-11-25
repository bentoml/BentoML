

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from .configuration_prophetnet import ProphetNetConfig

logger = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST = ...
PROPHETNET_START_DOCSTRING = ...
PROPHETNET_INPUTS_DOCSTRING = ...
PROPHETNET_STANDALONE_INPUTS_DOCSTRING = ...
def softmax(hidden_state, dim, onnx_trace=...):
    ...

def ngram_attention_bias(sequence_length, ngram, device, dtype):
    """
    This function computes the bias for the predict stream
    """
    ...

def compute_relative_buckets(num_buckets, max_distance, relative_positions, is_bidirectional=...):
    """
    This function computes individual parts of the relative position buckets. For more detail, see paper.
    """
    ...

def compute_all_stream_relative_buckets(num_buckets, max_distance, position_ids):
    """
    This function computes both main and predict relative position buckets. For more detail, see paper.
    """
    ...

@dataclass
class ProphetNetSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, decoder_sequence_length, config.vocab_size)`):
            Prediction scores of the main stream language modeling head (scores for each vocabulary token before
            SoftMax).
        logits_ngram (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, ngram * decoder_sequence_length, config.vocab_size)`):
            Prediction scores of the predict stream language modeling head (scores for each vocabulary token before
            SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`, with each tensor of shape :obj:`(2,
            batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see :obj:`past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, decoder_sequence_length, hidden_size)`.

            Hidden-states of main stream of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_ngram_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, ngram * decoder_sequence_length, hidden_size)`.

            Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
            outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_attn_heads,
            decoder_sequence_length, decoder_sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        decoder_ngram_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_attn_heads,
            decoder_sequence_length, decoder_sequence_length)`.

            Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_attn_heads,
            encoder_sequence_length, decoder_sequence_length)`.

            Attentions weights of the cross-attention layer of the decoder, after the attention softmax, used to
            compute the weighted average in the
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_attn_heads,
            encoder_sequence_length, encoder_sequence_length)`. Attentions weights of the encoder, after the attention
            softmax, used to compute the weighted average in the self-attention heads.
    """
    loss: Optional[torch.FloatTensor] = ...
    logits: torch.FloatTensor = ...
    logits_ngram: Optional[torch.FloatTensor] = ...
    past_key_values: Optional[Tuple[torch.FloatTensor]] = ...
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    decoder_ngram_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    decoder_ngram_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    encoder_last_hidden_state: Optional[torch.FloatTensor] = ...
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    @property
    def decoder_cross_attentions(self):
        ...
    


@dataclass
class ProphetNetSeq2SeqModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, decoder_sequence_length, hidden_size)`):
            Sequence of main stream hidden-states at the output of the last layer of the decoder of the model.

            If :obj:`past_key_values` is used only the last hidden-state of the sequences of shape :obj:`(batch_size,
            1, hidden_size)` is output.
        last_hidden_state_ngram (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,ngram * decoder_sequence_length, config.vocab_size)`):
            Sequence of predict stream hidden-states at the output of the last layer of the decoder of the model.
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`, with each tensor of shape :obj:`(2,
            batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see :obj:`past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, decoder_sequence_length, hidden_size)`.

            Hidden-states of main stream of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_ngram_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, ngram * decoder_sequence_length, hidden_size)`.

            Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
            outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_attn_heads,
            decoder_sequence_length, decoder_sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        decoder_ngram_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_attn_heads,
            decoder_sequence_length, decoder_sequence_length)`.

            Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
            weighted average in the
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_attn_heads,
            encoder_sequence_length, decoder_sequence_length)`.

            Attentions weights of the cross-attention layer of the decoder, after the attention softmax, used to
            compute the weighted average in the
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_attn_heads,
            encoder_sequence_length, encoder_sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """
    last_hidden_state: torch.FloatTensor
    last_hidden_state_ngram: Optional[torch.FloatTensor] = ...
    past_key_values: Optional[Tuple[torch.FloatTensor]] = ...
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    decoder_ngram_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    decoder_ngram_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    encoder_last_hidden_state: Optional[torch.FloatTensor] = ...
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    @property
    def decoder_cross_attentions(self):
        ...
    


@dataclass
class ProphetNetDecoderModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, decoder_sequence_length, hidden_size)`):
            Sequence of main stream hidden-states at the output of the last layer of the decoder of the model.

            If :obj:`past_key_values` is used only the last hidden-state of the sequences of shape :obj:`(batch_size,
            1, hidden_size)` is output.
        last_hidden_state_ngram (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, ngram * decoder_sequence_length, config.vocab_size)`):
            Sequence of predict stream hidden-states at the output of the last layer of the decoder of the model.
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`, with each tensor of shape :obj:`(2,
            batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see :obj:`past_key_values` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, decoder_sequence_length, hidden_size)`.

            Hidden-states of main stream of the decoder at the output of each layer plus the initial embedding outputs.
        ngram_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, ngram * decoder_sequence_length, hidden_size)`.

            Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
            outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_attn_heads,
            decoder_sequence_length, decoder_sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        ngram_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_attn_heads,
            decoder_sequence_length, decoder_sequence_length)`.

            Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
            weighted average in the
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_attn_heads,
            encoder_sequence_length, decoder_sequence_length)`.

            Attentions weights of the cross-attention layer of the decoder, after the attention softmax, used to
            compute the weighted average in the
    """
    last_hidden_state: torch.FloatTensor
    last_hidden_state_ngram: Optional[torch.FloatTensor] = ...
    past_key_values: Optional[Tuple[torch.FloatTensor]] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    hidden_states_ngram: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...
    ngram_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class ProphetNetDecoderLMOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, decoder_sequence_length, config.vocab_size)`):
            Prediction scores of the main stream language modeling head (scores for each vocabulary token before
            SoftMax).
        logits_ngram (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, ngram * decoder_sequence_length, config.vocab_size)`):
            Prediction scores of the predict stream language modeling head (scores for each vocabulary token before
            SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`, with each tensor of shape :obj:`(2,
            batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see :obj:`past_key_values` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, decoder_sequence_length, hidden_size)`.

            Hidden-states of main stream of the decoder at the output of each layer plus the initial embedding outputs.
        ngram_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, ngram * decoder_sequence_length, hidden_size)`.

            Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
            outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_attn_heads,
            decoder_sequence_length, decoder_sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        ngram_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_attn_heads,
            decoder_sequence_length, decoder_sequence_length)`.

            Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
            weighted average in the
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_attn_heads,
            encoder_sequence_length, decoder_sequence_length)`.

            Attentions weights of the cross-attention layer of the decoder, after the attention softmax, used to
            compute the weighted average in the
    """
    loss: Optional[torch.FloatTensor] = ...
    logits: torch.FloatTensor = ...
    logits_ngram: Optional[torch.FloatTensor] = ...
    past_key_values: Optional[Tuple[torch.FloatTensor]] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    hidden_states_ngram: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...
    ngram_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = ...


class ProphetNetPreTrainedModel(PreTrainedModel):
    config_class = ProphetNetConfig
    base_model_prefix = ...


class ProphetNetPositionalEmbeddings(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size. Padding ids are ignored by either offsetting
    based on padding_idx or by setting padding_idx to None and ensuring that the appropriate position ids are passed to
    the forward function.
    """
    def __init__(self, config: ProphetNetConfig) -> None:
        ...
    
    def forward(self, inputs_shape, device, attention_mask=..., past_key_values=..., position_ids=...):
        ...
    


class ProphetNetAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: ProphetNetConfig, num_attn_heads: int) -> None:
        ...
    
    def forward(self, hidden_states, key_value_states: Optional[Tensor] = ..., attention_mask: Optional[Tensor] = ..., layer_head_mask: Optional[Tensor] = ..., past_key_value: Optional[Tuple[Tensor]] = ..., output_attentions: bool = ...) -> Tuple[Tensor, Optional[Tensor]]:
        ...
    


class ProphetNetFeedForward(nn.Module):
    """
    This is the residual two feed-forward layer block based on the original Transformer implementation.
    """
    def __init__(self, config: ProphetNetConfig, ffn_dim: int) -> None:
        ...
    
    def forward(self, hidden_states):
        ...
    


class ProphetNetNgramSelfAttention(nn.Module):
    def __init__(self, config: ProphetNetConfig) -> None:
        ...
    
    def prepare_for_onnx_export_(self):
        ...
    
    def forward(self, hidden_states, past_key_value: Optional[Tuple[Tensor]] = ..., attention_mask=..., layer_head_mask=..., extended_predict_attention_mask=..., main_relative_position_buckets=..., predict_relative_position_buckets=..., position_ids=...):
        ...
    
    def get_main_relative_pos_embeddings(self, hidden_states, attn_weights, position_ids, main_relative_position_buckets):
        ...
    
    def get_predict_relative_pos_embeddings(self, hidden_states, attn_weights, position_ids, predict_relative_position_buckets):
        ...
    


class ProphetNetEncoderLayer(nn.Module):
    """
    Encoder block for Prophetnet
    """
    def __init__(self, config: ProphetNetConfig) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions: bool = ...):
        ...
    


class ProphetNetDecoderLayer(nn.Module):
    """
    Decoder block for Prophetnet
    """
    def __init__(self, config: ProphetNetConfig) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., encoder_hidden_states=..., encoder_attn_mask=..., layer_head_mask=..., cross_attn_layer_head_mask=..., extended_predict_attention_mask=..., main_relative_position_buckets=..., predict_relative_position_buckets=..., position_ids=..., past_key_value=..., use_cache: bool = ..., output_attentions: bool = ...):
        ...
    


@add_start_docstrings("The standalone encoder part of the ProphetNetModel.", PROPHETNET_START_DOCSTRING)
class ProphetNetEncoder(ProphetNetPreTrainedModel):
    r"""
    word_embeddings  (:obj:`torch.nn.Embeddings` of shape :obj:`(config.vocab_size, config.hidden_size)`, `optional`):
        The word embedding parameters. This can be used to initialize :class:`~transformers.ProphetNetEncoder` with
        pre-defined word embeddings instead of randomly initialized word embeddings.
    """
    def __init__(self, config: ProphetNetConfig, word_embeddings: nn.Embedding = ...) -> None:
        ...
    
    def get_input_embeddings(self):
        ...
    
    def set_input_embeddings(self, value):
        ...
    
    @add_start_docstrings_to_model_forward(PROPHETNET_STANDALONE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        Returns:

        Example::

            >>> from transformers import ProphetNetTokenizer, ProphetNetEncoder
            >>> import torch

            >>> tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
            >>> model = ProphetNetEncoder.from_pretrained('patrickvonplaten/prophetnet-large-uncased-standalone')
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> last_hidden_states = outputs.last_hidden_state
        """
        ...
    


@add_start_docstrings("The standalone decoder part of the ProphetNetModel.", PROPHETNET_START_DOCSTRING)
class ProphetNetDecoder(ProphetNetPreTrainedModel):
    r"""
    word_embeddings  (:obj:`torch.nn.Embeddings` of shape :obj:`(config.vocab_size, config.hidden_size)`, `optional`):
        The word embedding parameters. This can be used to initialize :class:`~transformers.ProphetNetEncoder` with
        pre-defined word embeddings instead of randomly initialized word embeddings.
    """
    def __init__(self, config: ProphetNetConfig, word_embeddings: nn.Embedding = ...) -> None:
        ...
    
    def get_input_embeddings(self):
        ...
    
    def set_input_embeddings(self, value):
        ...
    
    @add_start_docstrings_to_model_forward(PROPHETNET_STANDALONE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ProphetNetDecoderModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., head_mask=..., cross_attn_head_mask=..., past_key_values=..., inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
        cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
            Mask to nullify selected heads of the cross-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last ``decoder_input_ids``
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all ``decoder_input_ids`` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        Returns:

        Example::

            >>> from transformers import ProphetNetTokenizer, ProphetNetDecoder
            >>> import torch

            >>> tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
            >>> model = ProphetNetDecoder.from_pretrained('microsoft/prophetnet-large-uncased', add_cross_attention=False)
            >>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> last_hidden_states = outputs.last_hidden_state
        """
        ...
    
    def compute_buffered_relative_buckets(self, position_ids):
        ...
    
    def prepare_attention_mask(self, hidden_states, attention_mask):
        ...
    
    def prepare_predict_attention_mask(self, hidden_states, attention_mask):
        ...
    


@add_start_docstrings("The bare ProphetNet Model outputting raw hidden-states without any specific head on top.", PROPHETNET_START_DOCSTRING)
class ProphetNetModel(ProphetNetPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self):
        ...
    
    def set_input_embeddings(self, value):
        ...
    
    def get_encoder(self):
        ...
    
    def get_decoder(self):
        ...
    
    @add_start_docstrings_to_model_forward(PROPHETNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ProphetNetSeq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., decoder_input_ids=..., decoder_attention_mask=..., head_mask=..., decoder_head_mask=..., cross_attn_head_mask=..., encoder_outputs: Optional[Tuple] = ..., past_key_values=..., inputs_embeds=..., decoder_inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        Returns:

        Example::

            >>> from transformers import ProphetNetTokenizer, ProphetNetModel

            >>> tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
            >>> model = ProphetNetModel.from_pretrained('microsoft/prophetnet-large-uncased')

            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

            >>> last_hidden_states = outputs.last_hidden_state  # main stream hidden states
            >>> last_hidden_states_ngram = outputs.last_hidden_state_ngram  # predict hidden states
        """
        ...
    


@add_start_docstrings("The ProphetNet Model with a language modeling head. Can be used for sequence generation tasks.", PROPHETNET_START_DOCSTRING)
class ProphetNetForConditionalGeneration(ProphetNetPreTrainedModel):
    def __init__(self, config: ProphetNetConfig) -> None:
        ...
    
    def get_output_embeddings(self):
        ...
    
    def set_output_embeddings(self, new_embeddings):
        ...
    
    def get_input_embeddings(self):
        ...
    
    @add_start_docstrings_to_model_forward(PROPHETNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ProphetNetSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., decoder_input_ids=..., decoder_attention_mask=..., head_mask=..., decoder_head_mask=..., cross_attn_head_mask=..., encoder_outputs=..., past_key_values=..., inputs_embeds=..., decoder_inputs_embeds=..., labels=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Example::

            >>> from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration

            >>> tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
            >>> model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased')

            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

            >>> logits_next_token = outputs.logits  # logits to predict next token as usual
            >>> logits_ngram_next_tokens = outputs.logits_ngram  # logits to predict 2nd, 3rd, ... next tokens
        """
        ...
    
    def prepare_inputs_for_generation(self, decoder_input_ids, past=..., attention_mask=..., head_mask=..., decoder_head_mask=..., cross_attn_head_mask=..., use_cache=..., encoder_outputs=..., **kwargs):
        ...
    
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        ...
    
    def get_encoder(self):
        ...
    
    def get_decoder(self):
        ...
    


@add_start_docstrings("The standalone decoder part of the ProphetNetModel with a lm head on top. The model can be used for causal language modeling.", PROPHETNET_START_DOCSTRING)
class ProphetNetForCausalLM(ProphetNetPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self):
        ...
    
    def set_input_embeddings(self, value):
        ...
    
    def get_output_embeddings(self):
        ...
    
    def set_output_embeddings(self, new_embeddings):
        ...
    
    def set_decoder(self, decoder):
        ...
    
    def get_decoder(self):
        ...
    
    @add_start_docstrings_to_model_forward(PROPHETNET_STANDALONE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ProphetNetDecoderLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., head_mask=..., cross_attn_head_mask=..., past_key_values=..., inputs_embeds=..., labels=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
        cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
            Mask to nullify selected heads of the cross-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last ``decoder_input_ids``
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all ``decoder_input_ids`` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
            ignored (masked), the loss is only computed for the tokens with labels n ``[0, ..., config.vocab_size]``

        Returns:

        Example::

            >>> from transformers import ProphetNetTokenizer, ProphetNetForCausalLM
            >>> import torch

            >>> tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
            >>> model = ProphetNetForCausalLM.from_pretrained('microsoft/prophetnet-large-uncased')
            >>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> logits = outputs.logits

            >>> # Model can also be used with EncoderDecoder framework
            >>> from transformers import BertTokenizer, EncoderDecoderModel, ProphetNetTokenizer
            >>> import torch

            >>> tokenizer_enc = BertTokenizer.from_pretrained('bert-large-uncased')
            >>> tokenizer_dec = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
            >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-large-uncased", "microsoft/prophetnet-large-uncased")

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
        ...
    
    def prepare_inputs_for_generation(self, input_ids, past=..., attention_mask=..., head_mask=..., use_cache=..., **kwargs):
        ...
    


class ProphetNetDecoderWrapper(ProphetNetPreTrainedModel):
    """
    This is a wrapper class, so that :class:`~transformers.ProphetNetForCausalLM` can correctly be loaded from
    pretrained prophetnet classes.
    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, *args, **kwargs):
        ...
    


