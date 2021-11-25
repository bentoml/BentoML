

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf

from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    TFSharedEmbeddings,
    keras_serializable,
)
from .configuration_led import LEDConfig

""" TF 2.0 LED model. """
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
LARGE_NEGATIVE = ...
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    ...

class TFLEDLearnedPositionalEmbedding(TFSharedEmbeddings):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs) -> None:
        ...
    
    def call(self, input_shape: tf.TensorShape, past_key_values_length: int = ...):
        """Input is expected to be of size [bsz x seqlen]."""
        ...
    


class TFLEDEncoderSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, layer_id, **kwargs) -> None:
        ...
    
    def call(self, inputs, training=...): # -> tuple[Unknown, Unknown, Unknown]:
        """
        LongformerSelfAttention expects `len(hidden_states)` to be multiple of `attention_window`. Padding to
        `attention_window` happens in LongformerModel.forward to avoid redoing the padding on each layer.

        The `attention_mask` is changed in :meth:`LongformerModel.forward` from 0, 1, 2 to:

            * -10000: no attention
            * 0: local attention
            * +10000: global attention
        """
        ...
    
    def reshape_and_transpose(self, vector, batch_size):
        ...
    


class TFLEDEncoderAttention(tf.keras.layers.Layer):
    def __init__(self, config, layer_id, **kwargs) -> None:
        ...
    
    def call(self, inputs, training=...):
        ...
    


class TFLEDDecoderAttention(tf.keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = ..., is_decoder: bool = ..., bias: bool = ..., **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, key_value_states: Optional[tf.Tensor] = ..., past_key_value: Optional[Tuple[Tuple[tf.Tensor]]] = ..., attention_mask: Optional[tf.Tensor] = ..., layer_head_mask: Optional[tf.Tensor] = ..., training=...) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """Input shape: Batch x Time x Channel"""
        ...
    


class TFLEDEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: LEDConfig, layer_id: int, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, layer_head_mask: tf.Tensor, is_index_masked: tf.Tensor, is_index_global_attn: tf.Tensor, is_global_attn: bool, training=...):
        """
        Args:
            hidden_states (:obj:`tf.Tensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`tf.Tensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
        """
        ...
    


class TFLEDDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: LEDConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states, attention_mask: Optional[tf.Tensor] = ..., encoder_hidden_states: Optional[tf.Tensor] = ..., encoder_attention_mask: Optional[tf.Tensor] = ..., layer_head_mask: Optional[tf.Tensor] = ..., encoder_layer_head_mask: Optional[tf.Tensor] = ..., past_key_value: Optional[Tuple[tf.Tensor]] = ..., training=...) -> Tuple[tf.Tensor, tf.Tensor, Tuple[Tuple[tf.Tensor]]]:
        """
        Args:
            hidden_states (:obj:`tf.Tensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`tf.Tensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`tf.Tensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`tf.Tensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            encoder_layer_head_mask (:obj:`tf.Tensor`): mask for encoder attention heads in a given layer of
                size `(config.encoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(tf.Tensor)`): cached past key and value projection states
        """
        ...
    


class TFLEDPreTrainedModel(TFPreTrainedModel):
    config_class = LEDConfig
    base_model_prefix = ...
    @property
    def dummy_inputs(self): # -> dict[str, Unknown]:
        ...
    
    @tf.function(input_signature=[{ "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),"attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),"decoder_input_ids": tf.TensorSpec((None, None), tf.int32, name="decoder_input_ids"),"decoder_attention_mask": tf.TensorSpec((None, None), tf.int32, name="decoder_attention_mask") }])
    def serving(self, inputs):
        ...
    


@dataclass
class TFLEDEncoderBaseModelOutput(ModelOutput):
    """
    Base class for Longformer's outputs, with potential hidden states, local and global attentions.

    Args:
        last_hidden_state (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length, x +
            attention_window + 1)`, where ``x`` is the number of tokens with global attention mask.

            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first ``x`` values) and to every token in the attention window (remaining
            ``attention_window + 1`` values). Note that the first ``x`` values refer to tokens with fixed positions in
            the text, but the remaining ``attention_window + 1`` values refer to tokens with relative positions: the
            attention weight of a token to itself is located at index ``x + attention_window / 2`` and the
            ``attention_window / 2`` preceding (succeeding) values are the attention weights to the ``attention_window
            / 2`` preceding (succeeding) tokens. If the attention window contains a token with global attention, the
            attention weight at the corresponding index is set to 0; the value should be accessed from the first ``x``
            attention weights. If a token has global attention, the attention weights to all other tokens in
            :obj:`attentions` is set to 0, the values should be accessed from :obj:`global_attentions`.
        global_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length, x)`,
            where ``x`` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """
    last_hidden_state: tf.Tensor = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...
    global_attentions: Optional[Tuple[tf.Tensor]] = ...


@dataclass
class TFLEDSeq2SeqModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    Args:
        last_hidden_state (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.

            If :obj:`past_key_values` is used only the last hidden-state of the sequences of shape :obj:`(batch_size,
            1, hidden_size)` is output.
        past_key_values (:obj:`List[tf.Tensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`tf.Tensor` of length :obj:`config.n_layers`, with each tensor of shape :obj:`(2, batch_size,
            num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see :obj:`past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_global_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length, x)`,
            where ``x`` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """
    last_hidden_state: tf.Tensor = ...
    past_key_values: Optional[List[tf.Tensor]] = ...
    decoder_hidden_states: Optional[Tuple[tf.Tensor]] = ...
    decoder_attentions: Optional[Tuple[tf.Tensor]] = ...
    cross_attentions: Optional[Tuple[tf.Tensor]] = ...
    encoder_last_hidden_state: Optional[tf.Tensor] = ...
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = ...
    encoder_attentions: Optional[Tuple[tf.Tensor]] = ...
    encoder_global_attentions: Optional[Tuple[tf.Tensor]] = ...


@dataclass
class TFLEDSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (:obj:`tf.Tensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss.
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`List[tf.Tensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`tf.Tensor` of length :obj:`config.n_layers`, with each tensor of shape :obj:`(2, batch_size,
            num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see :obj:`past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_global_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length, x)`,
            where ``x`` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """
    loss: Optional[tf.Tensor] = ...
    logits: tf.Tensor = ...
    past_key_values: Optional[List[tf.Tensor]] = ...
    decoder_hidden_states: Optional[Tuple[tf.Tensor]] = ...
    decoder_attentions: Optional[Tuple[tf.Tensor]] = ...
    cross_attentions: Optional[Tuple[tf.Tensor]] = ...
    encoder_last_hidden_state: Optional[tf.Tensor] = ...
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = ...
    encoder_attentions: Optional[Tuple[tf.Tensor]] = ...
    encoder_global_attentions: Optional[Tuple[tf.Tensor]] = ...


LED_START_DOCSTRING = ...
LED_INPUTS_DOCSTRING = ...
@keras_serializable
class TFLEDEncoder(tf.keras.layers.Layer):
    config_class = LEDConfig
    def __init__(self, config: LEDConfig, embed_tokens: Optional[TFSharedEmbeddings] = ..., **kwargs) -> None:
        ...
    
    def get_embed_tokens(self): # -> TFSharedEmbeddings | None:
        ...
    
    def set_embed_tokens(self, embed_tokens): # -> None:
        ...
    
    def call(self, input_ids=..., inputs_embeds=..., attention_mask=..., global_attention_mask=..., head_mask=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        """
        Args:
            input_ids (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.LEDTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`tf.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        ...
    
    @tf.function
    def compute_hidden_states(self, hidden_states, padding_len):
        ...
    


@keras_serializable
class TFLEDDecoder(tf.keras.layers.Layer):
    config_class = LEDConfig
    def __init__(self, config: LEDConfig, embed_tokens: Optional[TFSharedEmbeddings] = ..., **kwargs) -> None:
        ...
    
    def set_embed_tokens(self, embed_tokens): # -> None:
        ...
    
    def call(self, input_ids=..., inputs_embeds=..., attention_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., head_mask=..., encoder_head_mask=..., past_key_values=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        r"""
        Args:
            input_ids (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it. Indices can be obtained using :class:`~transformers.LEDTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details. `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`tf.Tensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`tf.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            encoder_head_mask (:obj:`tf.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (:obj:`Tuple[Tuple[tf.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding. If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        ...
    


@keras_serializable
class TFLEDMainLayer(tf.keras.layers.Layer):
    config_class = LEDConfig
    def __init__(self, config: LEDConfig, **kwargs) -> None:
        ...
    
    def get_input_embeddings(self): # -> TFSharedEmbeddings:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    def call(self, input_ids=..., attention_mask=..., decoder_input_ids=..., decoder_attention_mask=..., head_mask=..., decoder_head_mask=..., encoder_outputs: Optional[Union[Tuple, TFLEDEncoderBaseModelOutput]] = ..., global_attention_mask=..., past_key_values=..., inputs_embeds=..., decoder_inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs): # -> TFLEDSeq2SeqModelOutput:
        ...
    


@add_start_docstrings("The bare LED Model outputting raw hidden-states without any specific head on top.", LED_START_DOCSTRING)
class TFLEDModel(TFLEDPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def get_encoder(self): # -> TFLEDEncoder:
        ...
    
    def get_decoder(self): # -> TFLEDDecoder:
        ...
    
    @add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFLEDSeq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., decoder_input_ids=..., decoder_attention_mask=..., head_mask=..., decoder_head_mask=..., encoder_outputs: Optional[Union[Tuple, TFLEDEncoderBaseModelOutput]] = ..., global_attention_mask=..., past_key_values=..., inputs_embeds=..., decoder_inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    
    def serving_output(self, output): # -> TFLEDSeq2SeqModelOutput:
        ...
    


@add_start_docstrings("The LED Model with a language modeling head. Can be used for summarization.", LED_START_DOCSTRING)
class TFLEDForConditionalGeneration(TFLEDPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def get_decoder(self): # -> TFLEDDecoder:
        ...
    
    def get_encoder(self): # -> TFLEDEncoder:
        ...
    
    def get_bias(self): # -> dict[str, Unknown]:
        ...
    
    def set_bias(self, value): # -> None:
        ...
    
    def get_output_embeddings(self):
        ...
    
    def set_output_embeddings(self, value): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFLEDSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., decoder_input_ids=..., decoder_attention_mask=..., head_mask=..., decoder_head_mask=..., encoder_outputs: Optional[TFLEDEncoderBaseModelOutput] = ..., global_attention_mask=..., past_key_values=..., inputs_embeds=..., decoder_inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs): # -> TFLEDSeq2SeqLMOutput:
        """
        Returns:

        Examples::

            >>> from transformers import LEDTokenizer, TFLEDForConditionalGeneration
            >>> import tensorflow as tf
            >>> mname = 'allenai/led-base-16384'
            >>> tokenizer = LEDTokenizer.from_pretrained(mname)
            >>> TXT = "My friends are <mask> but they eat too many carbs."
            >>> model = TFLEDForConditionalGeneration.from_pretrained(mname)
            >>> batch = tokenizer([TXT], return_tensors='tf')
            >>> logits = model(inputs=batch.input_ids).logits
            >>> probs = tf.nn.softmax(logits[0])
            >>> # probs[5] is associated with the mask token
        """
        ...
    
    def serving_output(self, output): # -> TFLEDSeq2SeqLMOutput:
        ...
    
    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, head_mask=..., use_cache=..., **kwargs) -> Dict:
        ...
    
    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        ...
    
    def compute_loss(self, labels, logits):
        """CrossEntropyLoss that ignores pad tokens"""
        ...
    


