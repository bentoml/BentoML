

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
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPooling,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFNextSentencePredictorOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFMultipleChoiceLoss,
    TFNextSentencePredictionLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    keras_serializable,
)
from .configuration_mobilebert import MobileBertConfig

""" TF 2.0 MobileBERT model. """
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
TF_MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class TFMobileBertIntermediate(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states):
        ...
    


class TFLayerNorm(tf.keras.layers.LayerNormalization):
    def __init__(self, feat_size, *args, **kwargs) -> None:
        ...
    


class TFNoNorm(tf.keras.layers.Layer):
    def __init__(self, feat_size, epsilon=..., **kwargs) -> None:
        ...
    
    def build(self, input_shape): # -> None:
        ...
    
    def call(self, inputs: tf.Tensor):
        ...
    


NORM2FN = ...
class TFMobileBertEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def build(self, input_shape): # -> None:
        ...
    
    def call(self, input_ids=..., position_ids=..., token_type_ids=..., inputs_embeds=..., training=...):
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (:obj:`tf.Tensor`): output embedding tensor.
        """
        ...
    


class TFMobileBertSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def transpose_for_scores(self, x, batch_size):
        ...
    
    def call(self, query_tensor, key_tensor, value_tensor, attention_mask, head_mask, output_attentions, training=...): # -> tuple[Unknown, Unknown] | tuple[Unknown]:
        ...
    


class TFMobileBertSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states, residual_tensor, training=...):
        ...
    


class TFMobileBertAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def prune_heads(self, heads):
        ...
    
    def call(self, query_tensor, key_tensor, value_tensor, layer_input, attention_mask, head_mask, output_attentions, training=...):
        ...
    


class TFOutputBottleneck(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states, residual_tensor, training=...):
        ...
    


class TFMobileBertOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states, residual_tensor_1, residual_tensor_2, training=...):
        ...
    


class TFBottleneckLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, inputs):
        ...
    


class TFBottleneck(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states): # -> Tuple[Unknown, ...] | tuple[Unknown, Unknown, Unknown, Unknown]:
        ...
    


class TFFFNOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states, residual_tensor):
        ...
    


class TFFFNLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states):
        ...
    


class TFMobileBertLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states, attention_mask, head_mask, output_attentions, training=...):
        ...
    


class TFMobileBertEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states, attention_mask, head_mask, output_attentions, output_hidden_states, return_dict, training=...):
        ...
    


class TFMobileBertPooler(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states):
        ...
    


class TFMobileBertPredictionHeadTransform(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states):
        ...
    


class TFMobileBertLMPredictionHead(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def build(self, input_shape): # -> None:
        ...
    
    def get_output_embeddings(self): # -> Self@TFMobileBertLMPredictionHead:
        ...
    
    def set_output_embeddings(self, value): # -> None:
        ...
    
    def get_bias(self): # -> dict[str, Unknown]:
        ...
    
    def set_bias(self, value): # -> None:
        ...
    
    def call(self, hidden_states):
        ...
    


class TFMobileBertMLMHead(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, sequence_output):
        ...
    


@keras_serializable
class TFMobileBertMainLayer(tf.keras.layers.Layer):
    config_class = MobileBertConfig
    def __init__(self, config, add_pooling_layer=..., **kwargs) -> None:
        ...
    
    def get_input_embeddings(self): # -> TFMobileBertEmbeddings:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs): # -> TFBaseModelOutputWithPooling:
        ...
    


class TFMobileBertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MobileBertConfig
    base_model_prefix = ...


@dataclass
class TFMobileBertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.TFMobileBertForPreTraining`.

    Args:
        prediction_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, 2)`):
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
    loss: Optional[tf.Tensor] = ...
    prediction_logits: tf.Tensor = ...
    seq_relationship_logits: tf.Tensor = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...


MOBILEBERT_START_DOCSTRING = ...
MOBILEBERT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare MobileBert Model transformer outputting raw hidden-states without any specific head on top.", MOBILEBERT_START_DOCSTRING)
class TFMobileBertModel(TFMobileBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    
    def serving_output(self, output: TFBaseModelOutputWithPooling) -> TFBaseModelOutputWithPooling:
        ...
    


@add_start_docstrings("""
    MobileBert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
    `next sentence prediction (classification)` head.
    """, MOBILEBERT_START_DOCSTRING)
class TFMobileBertForPreTraining(TFMobileBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def get_lm_head(self): # -> TFMobileBertLMPredictionHead:
        ...
    
    def get_prefix_bias_name(self):
        ...
    
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFMobileBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs): # -> TFMobileBertForPreTrainingOutput:
        r"""
        Return:

        Examples::

            >>> import tensorflow as tf
            >>> from transformers import MobileBertTokenizer, TFMobileBertForPreTraining

            >>> tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
            >>> model = TFMobileBertForPreTraining.from_pretrained('google/mobilebert-uncased')
            >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
            >>> outputs = model(input_ids)
            >>> prediction_scores, seq_relationship_scores = outputs[:2]

        """
        ...
    
    def serving_output(self, output): # -> TFMobileBertForPreTrainingOutput:
        ...
    


@add_start_docstrings("""MobileBert Model with a `language modeling` head on top. """, MOBILEBERT_START_DOCSTRING)
class TFMobileBertForMaskedLM(TFMobileBertPreTrainedModel, TFMaskedLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def get_lm_head(self): # -> TFMobileBertLMPredictionHead:
        ...
    
    def get_prefix_bias_name(self):
        ...
    
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs): # -> TFMaskedLMOutput:
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels
        """
        ...
    
    def serving_output(self, output: TFMaskedLMOutput) -> TFMaskedLMOutput:
        ...
    


class TFMobileBertOnlyNSPHead(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, pooled_output):
        ...
    


@add_start_docstrings("""MobileBert Model with a `next sentence prediction (classification)` head on top. """, MOBILEBERT_START_DOCSTRING)
class TFMobileBertForNextSentencePrediction(TFMobileBertPreTrainedModel, TFNextSentencePredictionLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFNextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., next_sentence_label=..., training=..., **kwargs): # -> TFNextSentencePredictorOutput:
        r"""
        Return:

        Examples::

            >>> import tensorflow as tf
            >>> from transformers import MobileBertTokenizer, TFMobileBertForNextSentencePrediction

            >>> tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
            >>> model = TFMobileBertForNextSentencePrediction.from_pretrained('google/mobilebert-uncased')

            >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
            >>> encoding = tokenizer(prompt, next_sentence, return_tensors='tf')

            >>> logits = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])[0]
        """
        ...
    
    def serving_output(self, output: TFNextSentencePredictorOutput) -> TFNextSentencePredictorOutput:
        ...
    


@add_start_docstrings("""
    MobileBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """, MOBILEBERT_START_DOCSTRING)
class TFMobileBertForSequenceClassification(TFMobileBertPreTrainedModel, TFSequenceClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    MobileBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, MOBILEBERT_START_DOCSTRING)
class TFMobileBertForQuestionAnswering(TFMobileBertPreTrainedModel, TFQuestionAnsweringLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    


@add_start_docstrings("""
    MobileBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """, MOBILEBERT_START_DOCSTRING)
class TFMobileBertForMultipleChoice(TFMobileBertPreTrainedModel, TFMultipleChoiceLoss):
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
    
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        ...
    
    @tf.function(input_signature=[{ "input_ids": tf.TensorSpec((None, None, None), tf.int32, name="input_ids"),"attention_mask": tf.TensorSpec((None, None, None), tf.int32, name="attention_mask"),"token_type_ids": tf.TensorSpec((None, None, None), tf.int32, name="token_type_ids") }])
    def serving(self, inputs: Dict[str, tf.Tensor]): # -> TFMultipleChoiceModelOutput:
        ...
    
    def serving_output(self, output: TFMultipleChoiceModelOutput) -> TFMultipleChoiceModelOutput:
        ...
    


@add_start_docstrings("""
    MobileBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """, MOBILEBERT_START_DOCSTRING)
class TFMobileBertForTokenClassification(TFMobileBertPreTrainedModel, TFTokenClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    


