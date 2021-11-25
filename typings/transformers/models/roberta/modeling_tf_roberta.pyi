

from typing import Optional, Tuple, Union

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
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    keras_serializable,
)
from .configuration_roberta import RobertaConfig

""" TF 2.0 RoBERTa model. """
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class TFRobertaEmbeddings(tf.keras.layers.Layer):
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
    


class TFRobertaPooler(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...
    


class TFRobertaSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs) -> None:
        ...
    
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool = ...) -> Tuple[tf.Tensor]:
        ...
    


class TFRobertaSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...
    


class TFRobertaAttention(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs) -> None:
        ...
    
    def prune_heads(self, heads):
        ...
    
    def call(self, input_tensor: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool = ...) -> Tuple[tf.Tensor]:
        ...
    


class TFRobertaIntermediate(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...
    


class TFRobertaOutput(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...
    


class TFRobertaLayer(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool = ...) -> Tuple[tf.Tensor]:
        ...
    


class TFRobertaEncoder(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool = ...) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        ...
    


@keras_serializable
class TFRobertaMainLayer(tf.keras.layers.Layer):
    config_class = RobertaConfig
    def __init__(self, config, add_pooling_layer=..., **kwargs) -> None:
        ...
    
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        ...
    
    def set_input_embeddings(self, value: tf.Variable): # -> None:
        ...
    
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: bool = ..., **kwargs) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        ...
    


class TFRobertaPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = RobertaConfig
    base_model_prefix = ...
    @tf.function(input_signature=[{ "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),"attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask") }])
    def serving(self, inputs):
        ...
    


ROBERTA_START_DOCSTRING = ...
ROBERTA_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.", ROBERTA_START_DOCSTRING)
class TFRobertaModel(TFRobertaPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    
    def serving_output(self, output: TFBaseModelOutputWithPooling) -> TFBaseModelOutputWithPooling:
        ...
    


class TFRobertaLMHead(tf.keras.layers.Layer):
    """Roberta Head for masked language modeling."""
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
    


@add_start_docstrings("""RoBERTa Model with a `language modeling` head on top. """, ROBERTA_START_DOCSTRING)
class TFRobertaForMaskedLM(TFRobertaPreTrainedModel, TFMaskedLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def get_lm_head(self): # -> TFRobertaLMHead:
        ...
    
    def get_prefix_bias_name(self):
        ...
    
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs): # -> TFMaskedLMOutput:
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        ...
    
    def serving_output(self, output: TFMaskedLMOutput) -> TFMaskedLMOutput:
        ...
    


class TFRobertaClassificationHead(tf.keras.layers.Layer):
    """Head for sentence-level classification tasks."""
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, features, training=...):
        ...
    


@add_start_docstrings("""
    RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """, ROBERTA_START_DOCSTRING)
class TFRobertaForSequenceClassification(TFRobertaPreTrainedModel, TFSequenceClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs): # -> TFSequenceClassifierOutput:
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    
    def serving_output(self, output: TFSequenceClassifierOutput) -> TFSequenceClassifierOutput:
        ...
    


@add_start_docstrings("""
    Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, ROBERTA_START_DOCSTRING)
class TFRobertaForMultipleChoice(TFRobertaPreTrainedModel, TFMultipleChoiceLoss):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @property
    def dummy_inputs(self): # -> dict[str, Unknown]:
        """
        Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        ...
    
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        ...
    
    @tf.function(input_signature=[{ "input_ids": tf.TensorSpec((None, None, None), tf.int32, name="input_ids"),"attention_mask": tf.TensorSpec((None, None, None), tf.int32, name="attention_mask") }])
    def serving(self, inputs): # -> TFMultipleChoiceModelOutput:
        ...
    
    def serving_output(self, output: TFMultipleChoiceModelOutput) -> TFMultipleChoiceModelOutput:
        ...
    


@add_start_docstrings("""
    RoBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, ROBERTA_START_DOCSTRING)
class TFRobertaForTokenClassification(TFRobertaPreTrainedModel, TFTokenClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs): # -> TFTokenClassifierOutput:
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        ...
    
    def serving_output(self, output: TFTokenClassifierOutput) -> TFTokenClassifierOutput:
        ...
    


@add_start_docstrings("""
    RoBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, ROBERTA_START_DOCSTRING)
class TFRobertaForQuestionAnswering(TFRobertaPreTrainedModel, TFQuestionAnsweringLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., start_positions=..., end_positions=..., training=..., **kwargs): # -> TFQuestionAnsweringModelOutput:
        r"""
        start_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        ...
    
    def serving_output(self, output: TFQuestionAnsweringModelOutput) -> TFQuestionAnsweringModelOutput:
        ...
    


