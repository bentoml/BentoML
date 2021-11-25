

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
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
from .configuration_albert import AlbertConfig

""" TF 2.0 ALBERT model. """
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class TFAlbertPreTrainingLoss:
    """
    Loss function suitable for ALBERT pretraining, that is, the task of pretraining a language model by combining SOP +
    MLM. .. note:: Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.
    """
    def compute_loss(self, labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        ...
    


class TFAlbertEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config: AlbertConfig, **kwargs) -> None:
        ...
    
    def build(self, input_shape: tf.TensorShape): # -> None:
        ...
    
    def call(self, input_ids: tf.Tensor = ..., position_ids: tf.Tensor = ..., token_type_ids: tf.Tensor = ..., inputs_embeds: tf.Tensor = ..., training: bool = ...) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (:obj:`tf.Tensor`): output embedding tensor.
        """
        ...
    


class TFAlbertAttention(tf.keras.layers.Layer):
    """Contains the complete attention sublayer, including both dropouts and layer norm."""
    def __init__(self, config: AlbertConfig, **kwargs) -> None:
        ...
    
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        ...
    
    def call(self, input_tensor: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool = ...) -> Tuple[tf.Tensor]:
        ...
    


class TFAlbertLayer(tf.keras.layers.Layer):
    def __init__(self, config: AlbertConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool = ...) -> Tuple[tf.Tensor]:
        ...
    


class TFAlbertLayerGroup(tf.keras.layers.Layer):
    def __init__(self, config: AlbertConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, output_hidden_states: bool, training: bool = ...) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        ...
    


class TFAlbertTransformer(tf.keras.layers.Layer):
    def __init__(self, config: AlbertConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool = ...) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        ...
    


class TFAlbertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = AlbertConfig
    base_model_prefix = ...


class TFAlbertMLMHead(tf.keras.layers.Layer):
    def __init__(self, config: AlbertConfig, input_embeddings: tf.keras.layers.Layer, **kwargs) -> None:
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
    


@keras_serializable
class TFAlbertMainLayer(tf.keras.layers.Layer):
    config_class = AlbertConfig
    def __init__(self, config: AlbertConfig, add_pooling_layer: bool = ..., **kwargs) -> None:
        ...
    
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        ...
    
    def set_input_embeddings(self, value: tf.Variable): # -> None:
        ...
    
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: bool = ..., **kwargs) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        ...
    


@dataclass
class TFAlbertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.TFAlbertForPreTraining`.

    Args:
        prediction_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        sop_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
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
    loss: tf.Tensor = ...
    prediction_logits: tf.Tensor = ...
    sop_logits: tf.Tensor = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...


ALBERT_START_DOCSTRING = ...
ALBERT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare Albert Model transformer outputting raw hidden-states without any specific head on top.", ALBERT_START_DOCSTRING)
class TFAlbertModel(TFAlbertPreTrainedModel):
    def __init__(self, config: AlbertConfig, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        ...
    
    def serving_output(self, output: TFBaseModelOutputWithPooling) -> TFBaseModelOutputWithPooling:
        ...
    


@add_start_docstrings("""
    Albert Model with two heads on top for pretraining: a `masked language modeling` head and a `sentence order
    prediction` (classification) head.
    """, ALBERT_START_DOCSTRING)
class TFAlbertForPreTraining(TFAlbertPreTrainedModel, TFAlbertPreTrainingLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: AlbertConfig, *inputs, **kwargs) -> None:
        ...
    
    def get_lm_head(self) -> tf.keras.layers.Layer:
        ...
    
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFAlbertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[Union[np.ndarray, tf.Tensor]] = ..., sentence_order_label: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFAlbertForPreTrainingOutput, Tuple[tf.Tensor]]:
        r"""
        Return:

        Example::

            >>> import tensorflow as tf
            >>> from transformers import AlbertTokenizer, TFAlbertForPreTraining

            >>> tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            >>> model = TFAlbertForPreTraining.from_pretrained('albert-base-v2')

            >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]  # Batch size 1
            >>> outputs = model(input_ids)

            >>> prediction_logits = outputs.prediction_logits
            >>> sop_logits = outputs.sop_logits
        """
        ...
    
    def serving_output(self, output: TFAlbertForPreTrainingOutput) -> TFAlbertForPreTrainingOutput:
        ...
    


class TFAlbertSOPHead(tf.keras.layers.Layer):
    def __init__(self, config: AlbertConfig, **kwargs) -> None:
        ...
    
    def call(self, pooled_output: tf.Tensor, training: bool) -> tf.Tensor:
        ...
    


@add_start_docstrings("""Albert Model with a `language modeling` head on top. """, ALBERT_START_DOCSTRING)
class TFAlbertForMaskedLM(TFAlbertPreTrainedModel, TFMaskedLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: AlbertConfig, *inputs, **kwargs) -> None:
        ...
    
    def get_lm_head(self) -> tf.keras.layers.Layer:
        ...
    
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        ...
    
    def serving_output(self, output: TFMaskedLMOutput) -> TFMaskedLMOutput:
        ...
    


@add_start_docstrings("""
    Albert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """, ALBERT_START_DOCSTRING)
class TFAlbertForSequenceClassification(TFAlbertPreTrainedModel, TFSequenceClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config: AlbertConfig, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in ``[0, ...,
            config.num_labels - 1]``. If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
        """
        ...
    
    def serving_output(self, output: TFSequenceClassifierOutput) -> TFSequenceClassifierOutput:
        ...
    


@add_start_docstrings("""
    Albert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, ALBERT_START_DOCSTRING)
class TFAlbertForTokenClassification(TFAlbertPreTrainedModel, TFTokenClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config: AlbertConfig, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        ...
    
    def serving_output(self, output: TFTokenClassifierOutput) -> TFTokenClassifierOutput:
        ...
    


@add_start_docstrings("""
    Albert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, ALBERT_START_DOCSTRING)
class TFAlbertForQuestionAnswering(TFAlbertPreTrainedModel, TFQuestionAnsweringLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: AlbertConfig, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., start_positions: Optional[Union[np.ndarray, tf.Tensor]] = ..., end_positions: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
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
    


@add_start_docstrings("""
    Albert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, ALBERT_START_DOCSTRING)
class TFAlbertForMultipleChoice(TFAlbertPreTrainedModel, TFMultipleChoiceLoss):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config: AlbertConfig, *inputs, **kwargs) -> None:
        ...
    
    @property
    def dummy_inputs(self): # -> dict[str, Unknown]:
        """
        Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        ...
    
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
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
    


