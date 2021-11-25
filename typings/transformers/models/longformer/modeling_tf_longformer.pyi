

from dataclasses import dataclass
from typing import Optional, Tuple

import tensorflow as tf

from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    keras_serializable,
)
from .configuration_longformer import LongformerConfig

"""Tensorflow Longformer model. """
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
TF_LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = ...
@dataclass
class TFLongformerBaseModelOutput(ModelOutput):
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
class TFLongformerBaseModelOutputWithPooling(ModelOutput):
    """
    Base class for Longformer's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`tf.Tensor` of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) further processed by a
            Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
            prediction (classification) objective during pretraining.
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
    pooler_output: tf.Tensor = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...
    global_attentions: Optional[Tuple[tf.Tensor]] = ...


@dataclass
class TFLongformerMaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        loss (:obj:`tf.Tensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Masked language modeling (MLM) loss.
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
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
    loss: Optional[tf.Tensor] = ...
    logits: tf.Tensor = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...
    global_attentions: Optional[Tuple[tf.Tensor]] = ...


@dataclass
class TFLongformerQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering Longformer models.

    Args:
        loss (:obj:`tf.Tensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
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
    loss: Optional[tf.Tensor] = ...
    start_logits: tf.Tensor = ...
    end_logits: tf.Tensor = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...
    global_attentions: Optional[Tuple[tf.Tensor]] = ...


@dataclass
class TFLongformerSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (:obj:`tf.Tensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
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
    loss: Optional[tf.Tensor] = ...
    logits: tf.Tensor = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...
    global_attentions: Optional[Tuple[tf.Tensor]] = ...


@dataclass
class TFLongformerMultipleChoiceModelOutput(ModelOutput):
    """
    Base class for outputs of multiple choice models.

    Args:
        loss (:obj:`tf.Tensor` of shape `(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss.
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, num_choices)`):
            `num_choices` is the second dimension of the input tensors. (see `input_ids` above).

            Classification scores (before SoftMax).
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
    loss: Optional[tf.Tensor] = ...
    logits: tf.Tensor = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...
    global_attentions: Optional[Tuple[tf.Tensor]] = ...


@dataclass
class TFLongformerTokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (:obj:`tf.Tensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
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
    loss: Optional[tf.Tensor] = ...
    logits: tf.Tensor = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...
    global_attentions: Optional[Tuple[tf.Tensor]] = ...


class TFLongformerLMHead(tf.keras.layers.Layer):
    """Longformer Head for masked language modeling."""
    def __init__(self, config, input_embeddings, **kwargs) -> None:
        ...
    
    def build(self, input_shape): # -> None:
        ...
    
    def get_output_embeddings(self):
        ...
    
    def set_output_embeddings(self, value): # -> None:
        ...
    
    def get_bias(self): # -> dict[str, Unknown]:
        ...
    
    def set_bias(self, value): # -> None:
        ...
    
    def call(self, hidden_states):
        ...
    


class TFLongformerEmbeddings(tf.keras.layers.Layer):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def build(self, input_shape: tf.TensorShape): # -> None:
        ...
    
    def create_position_ids_from_input_ids(self, input_ids):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            input_ids: tf.Tensor
        Returns: tf.Tensor
        """
        ...
    
    def call(self, input_ids=..., position_ids=..., token_type_ids=..., inputs_embeds=..., training=...):
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (:obj:`tf.Tensor`): output embedding tensor.
        """
        ...
    


class TFLongformerIntermediate(tf.keras.layers.Layer):
    def __init__(self, config: LongformerConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...
    


class TFLongformerOutput(tf.keras.layers.Layer):
    def __init__(self, config: LongformerConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...
    


class TFLongformerPooler(tf.keras.layers.Layer):
    def __init__(self, config: LongformerConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...
    


class TFLongformerSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config: LongformerConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...
    


class TFLongformerSelfAttention(tf.keras.layers.Layer):
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
    


class TFLongformerAttention(tf.keras.layers.Layer):
    def __init__(self, config, layer_id=..., **kwargs) -> None:
        ...
    
    def prune_heads(self, heads):
        ...
    
    def call(self, inputs, training=...):
        ...
    


class TFLongformerLayer(tf.keras.layers.Layer):
    def __init__(self, config, layer_id=..., **kwargs) -> None:
        ...
    
    def call(self, inputs, training=...):
        ...
    


class TFLongformerEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states, attention_mask=..., head_mask=..., padding_len=..., is_index_masked=..., is_index_global_attn=..., is_global_attn=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=...):
        ...
    


@keras_serializable
class TFLongformerMainLayer(tf.keras.layers.Layer):
    config_class = LongformerConfig
    def __init__(self, config, add_pooling_layer=..., **kwargs) -> None:
        ...
    
    def get_input_embeddings(self): # -> TFLongformerEmbeddings:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    def call(self, input_ids=..., attention_mask=..., head_mask=..., global_attention_mask=..., token_type_ids=..., position_ids=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    


class TFLongformerPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = LongformerConfig
    base_model_prefix = ...
    @property
    def dummy_inputs(self): # -> dict[str, Unknown]:
        ...
    
    @tf.function(input_signature=[{ "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),"attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask") }])
    def serving(self, inputs):
        ...
    


LONGFORMER_START_DOCSTRING = ...
LONGFORMER_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare Longformer Model outputting raw hidden-states without any specific head on top.", LONGFORMER_START_DOCSTRING)
class TFLongformerModel(TFLongformerPreTrainedModel):
    """

    This class copies code from :class:`~transformers.TFRobertaModel` and overwrites standard self-attention with
    longformer self-attention to provide the ability to process long sequences following the self-attention approach
    described in `Longformer: the Long-Document Transformer <https://arxiv.org/abs/2004.05150>`__ by Iz Beltagy,
    Matthew E. Peters, and Arman Cohan. Longformer self-attention combines a local (sliding window) and global
    attention to extend to long documents without the O(n^2) increase in memory and compute.

    The self-attention module :obj:`TFLongformerSelfAttention` implemented here supports the combination of local and
    global attention but it lacks support for autoregressive attention and dilated attention. Autoregressive and
    dilated attention are more relevant for autoregressive language modeling than finetuning on downstream tasks.
    Future release will add support for autoregressive attention, but the support for dilated attention requires a
    custom CUDA kernel to be memory and compute efficient.

    """
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def call(self, input_ids=..., attention_mask=..., head_mask=..., global_attention_mask=..., token_type_ids=..., position_ids=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    
    def serving_output(self, output): # -> TFLongformerBaseModelOutputWithPooling:
        ...
    


@add_start_docstrings("""Longformer Model with a `language modeling` head on top. """, LONGFORMER_START_DOCSTRING)
class TFLongformerForMaskedLM(TFLongformerPreTrainedModel, TFMaskedLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def get_lm_head(self): # -> TFLongformerLMHead:
        ...
    
    def get_prefix_bias_name(self):
        ...
    
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFLongformerMaskedLMOutput, config_class=_CONFIG_FOR_DOC, mask="<mask>")
    def call(self, input_ids=..., attention_mask=..., head_mask=..., global_attention_mask=..., token_type_ids=..., position_ids=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs): # -> TFLongformerMaskedLMOutput:
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        ...
    
    def serving_output(self, output): # -> TFLongformerMaskedLMOutput:
        ...
    


@add_start_docstrings("""
    Longformer Model with a span classification head on top for extractive question-answering tasks like SQuAD /
    TriviaQA (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, LONGFORMER_START_DOCSTRING)
class TFLongformerForQuestionAnswering(TFLongformerPreTrainedModel, TFQuestionAnsweringLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="allenai/longformer-large-4096-finetuned-triviaqa", output_type=TFLongformerQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., head_mask=..., global_attention_mask=..., token_type_ids=..., position_ids=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., start_positions=..., end_positions=..., training=..., **kwargs): # -> TFLongformerQuestionAnsweringModelOutput:
        r"""
        start_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        ...
    
    def serving_output(self, output): # -> TFLongformerQuestionAnsweringModelOutput:
        ...
    


class TFLongformerClassificationHead(tf.keras.layers.Layer):
    """Head for sentence-level classification tasks."""
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states, training=...):
        ...
    


@add_start_docstrings("""
    Longformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """, LONGFORMER_START_DOCSTRING)
class TFLongformerForSequenceClassification(TFLongformerPreTrainedModel, TFSequenceClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFLongformerSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., head_mask=..., token_type_ids=..., position_ids=..., global_attention_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs): # -> TFLongformerSequenceClassifierOutput:
        ...
    
    def serving_output(self, output): # -> TFLongformerSequenceClassifierOutput:
        ...
    


@add_start_docstrings("""
    Longformer Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """, LONGFORMER_START_DOCSTRING)
class TFLongformerForMultipleChoice(TFLongformerPreTrainedModel, TFMultipleChoiceLoss):
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @property
    def dummy_inputs(self): # -> dict[str, Unknown]:
        ...
    
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFLongformerMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., head_mask=..., token_type_ids=..., position_ids=..., global_attention_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        ...
    
    @tf.function(input_signature=[{ "input_ids": tf.TensorSpec((None, None, None), tf.int32, name="input_ids"),"attention_mask": tf.TensorSpec((None, None, None), tf.int32, name="attention_mask") }])
    def serving(self, inputs): # -> TFLongformerMultipleChoiceModelOutput:
        ...
    
    def serving_output(self, output): # -> TFLongformerMultipleChoiceModelOutput:
        ...
    


@add_start_docstrings("""
    Longformer Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """, LONGFORMER_START_DOCSTRING)
class TFLongformerForTokenClassification(TFLongformerPreTrainedModel, TFTokenClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFLongformerTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., head_mask=..., token_type_ids=..., position_ids=..., global_attention_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs): # -> TFLongformerTokenClassifierOutput:
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        ...
    
    def serving_output(self, output): # -> TFLongformerTokenClassifierOutput:
        ...
    


