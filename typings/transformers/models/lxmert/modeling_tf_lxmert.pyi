

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import tensorflow as tf

from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_utils import TFPreTrainedModel, keras_serializable
from .configuration_lxmert import LxmertConfig

""" TF 2.0 LXMERT model. """
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
TF_LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
@dataclass
class TFLxmertModelOutput(ModelOutput):
    """
    Lxmert's outputs that contain the last hidden states, pooled outputs, and attention probabilities for the language,
    visual, and, cross-modality encoders. (note: the visual encoder in Lxmert is referred to as the "relation-ship"
    encoder")


    Args:
        language_output (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the language encoder.
        vision_output (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the visual encoder.
        pooled_output (:obj:`tf.Tensor` of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification, CLS, token) further processed
            by a Linear layer and a Tanh activation function. The Linear
        language_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for input features + one for the output of each cross-modality layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for input features + one for the output of each cross-modality layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.
        language_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        vision_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_encoder_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """
    language_output: Optional[tf.Tensor] = ...
    vision_output: Optional[tf.Tensor] = ...
    pooled_output: Optional[tf.Tensor] = ...
    language_hidden_states: Optional[Tuple[tf.Tensor]] = ...
    vision_hidden_states: Optional[Tuple[tf.Tensor]] = ...
    language_attentions: Optional[Tuple[tf.Tensor]] = ...
    vision_attentions: Optional[Tuple[tf.Tensor]] = ...
    cross_encoder_attentions: Optional[Tuple[tf.Tensor]] = ...


@dataclass
class TFLxmertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.LxmertForPreTraining`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``tf.Tensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cross_relationship_score: (:obj:`tf.Tensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the textual matching objective (classification) head (scores of True/False
            continuation before SoftMax).
        question_answering_score: (:obj:`tf.Tensor` of shape :obj:`(batch_size, n_qa_answers)`):
            Prediction scores of question answering objective (classification).
        language_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for input features + one for the output of each cross-modality layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for input features + one for the output of each cross-modality layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.
        language_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        vision_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_encoder_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.

    """
    loss: Optional[tf.Tensor] = ...
    prediction_logits: Optional[tf.Tensor] = ...
    cross_relationship_score: Optional[tf.Tensor] = ...
    question_answering_score: Optional[tf.Tensor] = ...
    language_hidden_states: Optional[Tuple[tf.Tensor]] = ...
    vision_hidden_states: Optional[Tuple[tf.Tensor]] = ...
    language_attentions: Optional[Tuple[tf.Tensor]] = ...
    vision_attentions: Optional[Tuple[tf.Tensor]] = ...
    cross_encoder_attentions: Optional[Tuple[tf.Tensor]] = ...


class TFLxmertVisualFeatureEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, visn_input, training=...):
        ...
    


class TFLxmertEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def build(self, input_shape): # -> None:
        ...
    
    def call(self, input_ids=..., token_type_ids=..., inputs_embeds=..., training=...):
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (:obj:`tf.Tensor`): output embedding tensor.
        """
        ...
    


class TFLxmertAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def transpose_for_scores(self, x, batch_size):
        ...
    
    def call(self, hidden_states, context, attention_mask, output_attentions, training=...): # -> tuple[Unknown, Unknown] | tuple[Unknown]:
        ...
    


class TFLxmertIntermediate(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states):
        ...
    


class TFLxmertOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states, input_tensor, training=...):
        ...
    


class TFLxmertAttentionOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states, input_tensor, training=...):
        ...
    


class TFLxmertSelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, input_tensor, attention_mask, output_attentions, training=...): # -> tuple[Unknown, Unknown | Unbound] | tuple[Unknown]:
        ...
    


class TFLxmertCrossAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, input_tensor, ctx_tensor, ctx_att_mask, output_attentions=..., training=...): # -> tuple[Unknown, Unknown | Unbound] | tuple[Unknown]:
        ...
    


class TFLxmertLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states, attention_mask, output_attentions, training=...):
        ...
    


class TFLxmertXLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask, output_attentions, training=...): # -> tuple[Unknown, Unknown]:
        ...
    
    def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask, training=...): # -> tuple[Unknown, Unknown]:
        ...
    
    def output_fc(self, lang_input, visn_input, training=...): # -> tuple[Unknown, Unknown]:
        ...
    
    def call(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask, output_attentions, training=...): # -> tuple[Unknown, Unknown, Unknown] | tuple[Unknown, Unknown]:
        ...
    


class TFLxmertEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, lang_feats=..., lang_attention_mask=..., visual_feats=..., visual_pos=..., visual_attention_mask=..., output_attentions=..., training=...):
        ...
    


@keras_serializable
class TFLxmertMainLayer(tf.keras.layers.Layer):
    config_class = LxmertConfig
    @property
    def dummy_inputs(self): # -> dict[str, Unknown]:
        """
        Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        ...
    
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def get_input_embeddings(self): # -> TFLxmertEmbeddings:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    def call(self, input_ids=..., visual_feats=..., visual_pos=..., attention_mask=..., visual_attention_mask=..., token_type_ids=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    


class TFLxmertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = LxmertConfig
    base_model_prefix = ...
    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        ...
    
    @tf.function(input_signature=[{ "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),"attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),"visual_feats": tf.TensorSpec((None, None, None), tf.float32, name="visual_feats"),"visual_pos": tf.TensorSpec((None, None, None), tf.float32, name="visual_pos"),"visual_attention_mask": tf.TensorSpec((None, None), tf.int32, name="visual_attention_mask"),"token_type_ids": tf.TensorSpec((None, None), tf.int32, name="token_type_ids") }])
    def serving(self, inputs):
        ...
    


LXMERT_START_DOCSTRING = ...
LXMERT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare Lxmert Model transformer outputting raw hidden-states without any specific head on top.", LXMERT_START_DOCSTRING)
class TFLxmertModel(TFLxmertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFLxmertModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., visual_feats=..., visual_pos=..., attention_mask=..., visual_attention_mask=..., token_type_ids=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    
    def serving_output(self, output): # -> TFLxmertModelOutput:
        ...
    


class TFLxmertPooler(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states):
        ...
    


class TFLxmertPredictionHeadTransform(tf.keras.layers.Layer):
    def __init__(self, config: LxmertConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...
    


class TFLxmertLMPredictionHead(tf.keras.layers.Layer):
    def __init__(self, config: LxmertConfig, input_embeddings: tf.keras.layers.Layer, **kwargs) -> None:
        ...
    
    def build(self, input_shape: tf.TensorShape): # -> None:
        ...
    
    def get_output_embeddings(self) -> tf.keras.layers.Layer:
        ...
    
    def set_output_embeddings(self, value: tf.Variable): # -> None:
        ...
    
    def get_bias(self) -> Dict[str, tf.Variable]:
        ...
    
    def set_bias(self, value: tf.Variable): # -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...
    


class TFLxmertMLMHead(tf.keras.layers.Layer):
    def __init__(self, config: LxmertConfig, input_embeddings: tf.keras.layers.Layer, **kwargs) -> None:
        ...
    
    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        ...
    


class TFLxmertPreTrainingHeads(tf.keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs) -> None:
        ...
    
    def call(self, sequence_output, pooled_output): # -> tuple[Unknown, Unknown]:
        ...
    


class TFLxmertVisualAnswerHead(tf.keras.layers.Layer):
    def __init__(self, config, num_labels, **kwargs) -> None:
        ...
    
    def call(self, hidden_states):
        ...
    


class TFLxmertVisualObjHead(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states): # -> dict[Unknown, Unknown]:
        ...
    


@add_start_docstrings("""Lxmert Model with a `language modeling` head on top. """, LXMERT_START_DOCSTRING)
class TFLxmertForPreTraining(TFLxmertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @property
    def dummy_inputs(self): # -> dict[str, Unknown | Unbound | dict[Unknown, Unknown]]:
        """
        Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        ...
    
    def get_lm_head(self): # -> TFLxmertLMPredictionHead:
        ...
    
    def get_prefix_bias_name(self):
        ...
    
    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFLxmertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., visual_feats=..., visual_pos=..., attention_mask=..., visual_attention_mask=..., token_type_ids=..., inputs_embeds=..., masked_lm_labels=..., obj_labels=..., matched_label=..., ans=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        r"""
        masked_lm_labels (``tf.Tensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        obj_labels: (``Dict[Str: Tuple[tf.Tensor, tf.Tensor]]``, `optional`, defaults to :obj: `None`):
            each key is named after each one of the visual losses and each element of the tuple is of the shape
            ``(batch_size, num_features)`` and ``(batch_size, num_features, visual_feature_dim)`` for each the label id
            and the label score respectively
        matched_label (``tf.Tensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the whether or not the text input matches the image (classification) loss. Input
            should be a sequence pair (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:

            - 0 indicates that the sentence does not match the image,
            - 1 indicates that the sentence does match the image.
        ans: (``Torch.Tensor`` of shape ``(batch_size)``, `optional`, defaults to :obj: `None`):
            a one hot representation hof the correct answer `optional`

        Returns:
        """
        ...
    
    def serving_output(self, output): # -> TFLxmertForPreTrainingOutput:
        ...
    


