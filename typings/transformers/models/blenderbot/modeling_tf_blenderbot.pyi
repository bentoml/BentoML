

import os
from typing import Dict, Optional, Tuple, Union

import tensorflow as tf

from ...file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFSeq2SeqLMOutput,
    TFSeq2SeqModelOutput,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFPreTrainedModel,
    TFSharedEmbeddings,
    keras_serializable,
)
from .configuration_blenderbot import BlenderbotConfig

""" TF 2.0 Blenderbot model. """
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
LARGE_NEGATIVE = ...
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    ...

class TFBlenderbotLearnedPositionalEmbedding(TFSharedEmbeddings):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs) -> None:
        ...
    
    def call(self, input_shape: tf.TensorShape, past_key_values_length: int = ...):
        """Input is expected to be of size [bsz x seqlen]."""
        ...
    


class TFBlenderbotAttention(tf.keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = ..., is_decoder: bool = ..., bias: bool = ..., **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, key_value_states: Optional[tf.Tensor] = ..., past_key_value: Optional[Tuple[Tuple[tf.Tensor]]] = ..., attention_mask: Optional[tf.Tensor] = ..., layer_head_mask: Optional[tf.Tensor] = ..., training=...) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """Input shape: Batch x Time x Channel"""
        ...
    


class TFBlenderbotEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: BlenderbotConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, layer_head_mask: tf.Tensor, training=...): # -> tuple[Unknown, Unknown]:
        """
        Args:
            hidden_states (:obj:`tf.Tensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`tf.Tensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`
        """
        ...
    


class TFBlenderbotDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: BlenderbotConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states, attention_mask: Optional[tf.Tensor] = ..., encoder_hidden_states: Optional[tf.Tensor] = ..., encoder_attention_mask: Optional[tf.Tensor] = ..., layer_head_mask: Optional[tf.Tensor] = ..., cross_attn_layer_head_mask: Optional[tf.Tensor] = ..., past_key_value: Optional[Tuple[tf.Tensor]] = ..., training=...) -> Tuple[tf.Tensor, tf.Tensor, Tuple[Tuple[tf.Tensor]]]:
        """
        Args:
            hidden_states (:obj:`tf.Tensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`tf.Tensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`tf.Tensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`tf.Tensor`): mask for attention heads in a given layer of size
                `(decoder_attention_heads,)`
            cross_attn_layer_head_mask (:obj:`tf.Tensor`): mask for heads of the cross-attention module.
                `(decoder_attention_heads,)`
            past_key_value (:obj:`Tuple(tf.Tensor)`): cached past key and value projection states
        """
        ...
    


class TFBlenderbotPreTrainedModel(TFPreTrainedModel):
    config_class = BlenderbotConfig
    base_model_prefix = ...
    @property
    def dummy_inputs(self): # -> dict[str, Unknown]:
        ...
    
    @tf.function(input_signature=[{ "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),"attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),"decoder_input_ids": tf.TensorSpec((None, None), tf.int32, name="decoder_input_ids"),"decoder_attention_mask": tf.TensorSpec((None, None), tf.int32, name="decoder_attention_mask") }])
    def serving(self, inputs):
        ...
    


BLENDERBOT_START_DOCSTRING = ...
BLENDERBOT_GENERATION_EXAMPLE = ...
BLENDERBOT_INPUTS_DOCSTRING = ...
@keras_serializable
class TFBlenderbotEncoder(tf.keras.layers.Layer):
    config_class = BlenderbotConfig
    def __init__(self, config: BlenderbotConfig, embed_tokens: Optional[TFSharedEmbeddings] = ..., **kwargs) -> None:
        ...
    
    def get_embed_tokens(self): # -> TFSharedEmbeddings | None:
        ...
    
    def set_embed_tokens(self, embed_tokens): # -> None:
        ...
    
    def call(self, input_ids=..., inputs_embeds=..., attention_mask=..., head_mask=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        """
        Args:
            input_ids (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BlenderbotTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`tf.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value
                in the config will be used instead.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail. This argument can be used only in eager mode, in graph mode the value in the config
                will be used instead.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple. This
                argument can be used in eager mode, in graph mode the value will always be set to True.
            training (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use the model in training mode (some modules like dropout modules have different
                behaviors between training and evaluation).
        """
        ...
    


@keras_serializable
class TFBlenderbotDecoder(tf.keras.layers.Layer):
    config_class = BlenderbotConfig
    def __init__(self, config: BlenderbotConfig, embed_tokens: Optional[TFSharedEmbeddings] = ..., **kwargs) -> None:
        ...
    
    def get_embed_tokens(self): # -> TFSharedEmbeddings | None:
        ...
    
    def set_embed_tokens(self, embed_tokens): # -> None:
        ...
    
    def call(self, input_ids=..., inputs_embeds=..., attention_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., head_mask=..., cross_attn_head_mask=..., past_key_values=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        r"""
        Args:
            input_ids (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BlenderbotTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
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

            cross_attn_head_mask (:obj:`tf.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the cross-attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (:obj:`Tuple[Tuple[tf.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value
                in the config will be used instead.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail. This argument can be used only in eager mode, in graph mode the value in the config
                will be used instead.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple. This
                argument can be used in eager mode, in graph mode the value will always be set to True.
            training (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use the model in training mode (some modules like dropout modules have different
                behaviors between training and evaluation).
        """
        ...
    


@keras_serializable
class TFBlenderbotMainLayer(tf.keras.layers.Layer):
    config_class = BlenderbotConfig
    def __init__(self, config: BlenderbotConfig, **kwargs) -> None:
        ...
    
    def get_input_embeddings(self): # -> TFSharedEmbeddings:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    def call(self, input_ids=..., attention_mask=..., decoder_input_ids=..., decoder_attention_mask=..., head_mask=..., decoder_head_mask=..., cross_attn_head_mask=..., encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]] = ..., past_key_values=..., inputs_embeds=..., decoder_inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs): # -> TFSeq2SeqModelOutput:
        ...
    


@add_start_docstrings("The bare BLENDERBOT Model outputting raw hidden-states without any specific head on top.", BLENDERBOT_START_DOCSTRING)
class TFBlenderbotModel(TFBlenderbotPreTrainedModel):
    def __init__(self, config: BlenderbotConfig, *inputs, **kwargs) -> None:
        ...
    
    def get_encoder(self): # -> TFBlenderbotEncoder:
        ...
    
    def get_decoder(self): # -> TFBlenderbotDecoder:
        ...
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs): # -> tuple[Self@TFPreTrainedModel, dict[str, list[Any] | list[Unknown]]] | Self@TFPreTrainedModel:
        ...
    
    @add_start_docstrings_to_model_forward(BLENDERBOT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSeq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., decoder_input_ids=..., decoder_attention_mask=..., head_mask=..., decoder_head_mask=..., cross_attn_head_mask=..., encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]] = ..., past_key_values=..., inputs_embeds=..., decoder_inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    
    def serving_output(self, output): # -> TFSeq2SeqModelOutput:
        ...
    


@add_start_docstrings("The BLENDERBOT Model with a language modeling head. Can be used for summarization.", BLENDERBOT_START_DOCSTRING)
class TFBlenderbotForConditionalGeneration(TFBlenderbotPreTrainedModel, TFCausalLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def get_decoder(self): # -> TFBlenderbotDecoder:
        ...
    
    def get_encoder(self): # -> TFBlenderbotEncoder:
        ...
    
    def get_output_embeddings(self):
        ...
    
    def set_output_embeddings(self, value): # -> None:
        ...
    
    def get_bias(self): # -> dict[str, Unknown]:
        ...
    
    def set_bias(self, value): # -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs): # -> tuple[Self@TFPreTrainedModel, dict[str, list[Any] | list[Unknown]]] | Self@TFPreTrainedModel:
        ...
    
    @add_start_docstrings_to_model_forward(BLENDERBOT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BLENDERBOT_GENERATION_EXAMPLE)
    def call(self, input_ids=..., attention_mask=..., decoder_input_ids=..., decoder_attention_mask=..., head_mask=..., decoder_head_mask=..., cross_attn_head_mask=..., encoder_outputs: Optional[TFBaseModelOutput] = ..., past_key_values=..., inputs_embeds=..., decoder_inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs): # -> TFSeq2SeqLMOutput:
        r"""
        labels (:obj:`tf.tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        """
        ...
    
    def serving_output(self, output): # -> TFSeq2SeqLMOutput:
        ...
    
    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, head_mask=..., decoder_head_mask=..., cross_attn_head_mask=..., use_cache=..., **kwargs) -> Dict:
        ...
    


