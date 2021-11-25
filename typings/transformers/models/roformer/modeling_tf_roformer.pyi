

from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFCausalLMOutput,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    keras_serializable,
)
from .configuration_roformer import RoFormerConfig

""" TF 2.0 RoFormer model. """
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
TF_ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class TFRoFormerSinusoidalPositionalEmbedding(tf.keras.layers.Layer):
    """This module produces sinusoidal positional embeddings of any length."""
    def __init__(self, num_positions: int, embedding_dim: int, **kwargs) -> None:
        ...
    
    def build(self, input_shape: tf.TensorShape): # -> None:
        """
        Build shared token embedding layer Shared weights logic adapted from
        https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        ...
    
    def call(self, input_shape: tf.TensorShape, past_key_values_length: int = ...):
        """Input is expected to be of size [bsz x seqlen]."""
        ...
    


class TFRoFormerEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config: RoFormerConfig, **kwargs) -> None:
        ...
    
    def build(self, input_shape: tf.TensorShape): # -> None:
        ...
    
    def call(self, input_ids: tf.Tensor = ..., token_type_ids: tf.Tensor = ..., inputs_embeds: tf.Tensor = ..., training: bool = ...) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.


        Returns:
            final_embeddings (:obj:`tf.Tensor`): output embedding tensor.
        """
        ...
    


class TFRoFormerSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs) -> None:
        ...
    
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, sinusoidal_pos: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool = ...) -> Tuple[tf.Tensor]:
        ...
    
    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=...): # -> tuple[Unknown, Unknown, Unknown] | tuple[Unknown, Unknown]:
        ...
    


class TFRoFormerSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...
    


class TFRoFormerAttention(tf.keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs) -> None:
        ...
    
    def prune_heads(self, heads):
        ...
    
    def call(self, input_tensor: tf.Tensor, attention_mask: tf.Tensor, sinusoidal_pos: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool = ...) -> Tuple[tf.Tensor]:
        ...
    


class TFRoFormerIntermediate(tf.keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...
    


class TFRoFormerOutput(tf.keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...
    


class TFRoFormerLayer(tf.keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, sinusoidal_pos: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool = ...) -> Tuple[tf.Tensor]:
        ...
    


class TFRoFormerEncoder(tf.keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool = ...) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        ...
    


class TFRoFormerPredictionHeadTransform(tf.keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...
    


class TFRoFormerLMPredictionHead(tf.keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, input_embeddings: tf.keras.layers.Layer, **kwargs) -> None:
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
    


class TFRoFormerMLMHead(tf.keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, input_embeddings: tf.keras.layers.Layer, **kwargs) -> None:
        ...
    
    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        ...
    


@keras_serializable
class TFRoFormerMainLayer(tf.keras.layers.Layer):
    config_class = RoFormerConfig
    def __init__(self, config: RoFormerConfig, add_pooling_layer: bool = ..., **kwargs) -> None:
        ...
    
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        ...
    
    def set_input_embeddings(self, value: tf.Variable): # -> None:
        ...
    
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: bool = ..., **kwargs) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        ...
    


class TFRoFormerPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = RoFormerConfig
    base_model_prefix = ...


ROFORMER_START_DOCSTRING = ...
ROFORMER_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare RoFormer Model transformer outputing raw hidden-states without any specific head on top.", ROFORMER_START_DOCSTRING)
class TFRoFormerModel(TFRoFormerPreTrainedModel):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        ...
    
    def serving_output(self, output: TFBaseModelOutput) -> TFBaseModelOutput:
        ...
    


@add_start_docstrings("""RoFormer Model with a `language modeling` head on top. """, ROFORMER_START_DOCSTRING)
class TFRoFormerForMaskedLM(TFRoFormerPreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs) -> None:
        ...
    
    def get_lm_head(self) -> tf.keras.layers.Layer:
        ...
    
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (:obj:`tf.Tensor` or :obj:`np.ndarray` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        ...
    
    def serving_output(self, output: TFMaskedLMOutput) -> TFMaskedLMOutput:
        ...
    


@add_start_docstrings("""RoFormer Model with a `language modeling` head on top for CLM fine-tuning. """, ROFORMER_START_DOCSTRING)
class TFRoFormerForCausalLM(TFRoFormerPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs) -> None:
        ...
    
    def get_lm_head(self) -> tf.keras.layers.Layer:
        ...
    
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFCausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFCausalLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (:obj:`tf.Tensor` or :obj:`np.ndarray` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss. Indices should be in ``[0, ...,
            config.vocab_size - 1]``.
        """
        ...
    
    def serving_output(self, output: TFCausalLMOutput) -> TFCausalLMOutput:
        ...
    


class TFRoFormerClassificationHead(tf.keras.layers.Layer):
    """Head for sentence-level classification tasks."""
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...
    


@add_start_docstrings("""
    RoFormer Model transformer with a sequence classification/regression head on top e.g., for GLUE tasks.
    """, ROFORMER_START_DOCSTRING)
class TFRoFormerForSequenceClassification(TFRoFormerPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (:obj:`tf.Tensor` or :obj:`np.ndarray` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    
    def serving_output(self, output: TFSequenceClassifierOutput) -> TFSequenceClassifierOutput:
        ...
    


@add_start_docstrings("""
    RoFormer Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, ROFORMER_START_DOCSTRING)
class TFRoFormerForMultipleChoice(TFRoFormerPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs) -> None:
        ...
    
    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network.


        Returns:
            tf.Tensor with dummy inputs
        """
        ...
    
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (:obj:`tf.Tensor` or :obj:`np.ndarray` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        ...
    
    @tf.function(input_signature=[{ "input_ids": tf.TensorSpec((None, None, None), tf.int32, name="input_ids"),"attention_mask": tf.TensorSpec((None, None, None), tf.int32, name="attention_mask"),"token_type_ids": tf.TensorSpec((None, None, None), tf.int32, name="token_type_ids") }])
    def serving(self, inputs: Dict[str, tf.Tensor]) -> TFMultipleChoiceModelOutput:
        ...
    
    def serving_output(self, output: TFMultipleChoiceModelOutput) -> TFMultipleChoiceModelOutput:
        ...
    


@add_start_docstrings("""
    RoFormer Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, ROFORMER_START_DOCSTRING)
class TFRoFormerForTokenClassification(TFRoFormerPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (:obj:`tf.Tensor` or :obj:`np.ndarray` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        ...
    
    def serving_output(self, output: TFTokenClassifierOutput) -> TFTokenClassifierOutput:
        ...
    


@add_start_docstrings("""
    RoFormer Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, ROFORMER_START_DOCSTRING)
class TFRoFormerForQuestionAnswering(TFRoFormerPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., start_positions: Optional[Union[np.ndarray, tf.Tensor]] = ..., end_positions: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        r"""
        start_positions (:obj:`tf.Tensor` or :obj:`np.ndarray` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`tf.Tensor` or :obj:`np.ndarray` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        ...
    
    def serving_output(self, output: TFQuestionAnsweringModelOutput) -> TFQuestionAnsweringModelOutput:
        ...
    


