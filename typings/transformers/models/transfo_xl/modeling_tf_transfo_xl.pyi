

from dataclasses import dataclass
from typing import List, Optional, Tuple

import tensorflow as tf

from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras_serializable,
)
from .configuration_transfo_xl import TransfoXLConfig

"""
 TF 2.0 Transformer XL model.
"""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class TFPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, demb, **kwargs) -> None:
        ...
    
    def call(self, pos_seq, bsz=...):
        ...
    


class TFPositionwiseFF(tf.keras.layers.Layer):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=..., layer_norm_epsilon=..., init_std=..., **kwargs) -> None:
        ...
    
    def call(self, inp, training=...):
        ...
    


class TFRelPartialLearnableMultiHeadAttn(tf.keras.layers.Layer):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=..., pre_lnorm=..., r_r_bias=..., r_w_bias=..., layer_norm_epsilon=..., init_std=..., output_attentions=..., **kwargs) -> None:
        ...
    
    def build(self, input_shape): # -> None:
        ...
    
    def call(self, w, r, attn_mask, mems, head_mask, output_attentions, training=...): # -> list[Unknown]:
        ...
    


class TFRelPartialLearnableDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, dropatt=..., pre_lnorm=..., r_w_bias=..., r_r_bias=..., layer_norm_epsilon=..., init_std=..., output_attentions=..., **kwargs) -> None:
        ...
    
    def call(self, dec_inp, r, dec_attn_mask, mems, head_mask, output_attentions, training=...):
        ...
    


class TFTransfoEmbeddings(tf.keras.layers.Layer):
    def __init__(self, vocab_size, emb_size, init_std, **kwargs) -> None:
        ...
    
    def build(self, input_shape): # -> None:
        ...
    
    def call(self, inputs):
        ...
    


class TFAdaptiveEmbedding(tf.keras.layers.Layer):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=..., init_std=..., sample_softmax=..., **kwargs) -> None:
        ...
    
    def build(self, input_shape): # -> None:
        ...
    
    def call(self, inp):
        ...
    


@keras_serializable
class TFTransfoXLMainLayer(tf.keras.layers.Layer):
    config_class = TransfoXLConfig
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def build(self, input_shape): # -> None:
        ...
    
    def get_input_embeddings(self): # -> TFAdaptiveEmbedding:
        ...
    
    def set_input_embeddings(self, value):
        ...
    
    def backward_compatible(self): # -> None:
        ...
    
    def reset_memory_length(self, mem_len): # -> None:
        ...
    
    def init_mems(self, bsz): # -> list[Unknown] | None:
        ...
    
    def call(self, input_ids=..., mems=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    


class TFTransfoXLPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = TransfoXLConfig
    base_model_prefix = ...
    @tf.function(input_signature=[{ "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids") }])
    def serving(self, inputs):
        ...
    


@dataclass
class TFTransfoXLModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mems (:obj:`List[tf.Tensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see :obj:`mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    last_hidden_state: tf.Tensor = ...
    mems: List[tf.Tensor] = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...


@dataclass
class TFTransfoXLLMHeadModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        losses (:obj:`tf.Tensor` of shape `(batch_size, sequence_length-1)`, `optional`, returned when ``labels`` is provided)
            Language modeling losses (not reduced).
        prediction_scores (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token after SoftMax).
        mems (:obj:`List[tf.Tensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see :obj:`mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    prediction_scores: tf.Tensor = ...
    mems: List[tf.Tensor] = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...


@dataclass
class TFTransfoXLSequenceClassifierOutputWithPast(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (:obj:`tf.Tensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        mems (:obj:`List[tf.Tensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see :obj:`mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[tf.Tensor] = ...
    logits: tf.Tensor = ...
    mems: List[tf.Tensor] = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...


TRANSFO_XL_START_DOCSTRING = ...
TRANSFO_XL_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare Bert Model transformer outputting raw hidden-states without any specific head on top.", TRANSFO_XL_START_DOCSTRING)
class TFTransfoXLModel(TFTransfoXLPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTransfoXLModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., mems=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    
    def serving_output(self, output): # -> TFTransfoXLModelOutput:
        ...
    


@add_start_docstrings("""
    The Transformer-XL Model with a language modeling head on top (adaptive softmax with weights tied to the adaptive
    input embeddings)
    """, TRANSFO_XL_START_DOCSTRING)
class TFTransfoXLLMHeadModel(TFTransfoXLPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> None:
        """Double-check if you are using adaptive softmax."""
        ...
    
    def reset_memory_length(self, mem_len): # -> None:
        ...
    
    def init_mems(self, bsz): # -> list[Unknown] | None:
        ...
    
    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTransfoXLLMHeadModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., mems=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs): # -> TFTransfoXLLMHeadModelOutput:
        ...
    
    def serving_output(self, output): # -> TFTransfoXLLMHeadModelOutput:
        ...
    
    def prepare_inputs_for_generation(self, inputs, past, **model_kwargs): # -> dict[str, Unknown]:
        ...
    


@add_start_docstrings("""
    The Transfo XL Model transformer with a sequence classification head on top (linear layer).

    :class:`~transformers.TFTransfoXLForSequenceClassification` uses the last token in order to do the classification,
    as other causal models (e.g. GPT-1,GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    :obj:`pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each
    row. If no :obj:`pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot
    guess the padding tokens when :obj:`inputs_embeds` are passed instead of :obj:`input_ids`, it does the same (take
    the last value in each row of the batch).
    """, TRANSFO_XL_START_DOCSTRING)
class TFTransfoXLForSequenceClassification(TFTransfoXLPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def get_output_embeddings(self): # -> TFAdaptiveEmbedding:
        ...
    
    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTransfoXLSequenceClassifierOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., mems=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss. Indices should be in ``[0, ...,
            config.vocab_size - 1]``.
        """
        ...
    
    def serving_output(self, output): # -> TFTransfoXLSequenceClassifierOutputWithPast:
        ...
    


