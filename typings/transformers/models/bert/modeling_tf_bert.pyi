

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
    TFCausalLMOutput,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFNextSentencePredictorOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFNextSentencePredictionLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    keras_serializable,
)
from .configuration_bert import BertConfig

""" TF 2.0 BERT model. """
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class TFBertPreTrainingLoss:
    """
    Loss function suitable for BERT-like pretraining, that is, the task of pretraining a language model by combining
    NSP + MLM. .. note:: Any label of -100 will be ignored (along with the corresponding logits) in the loss
    computation.
    """
    def compute_loss(self, labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        ...
    


class TFBertEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config: BertConfig, **kwargs) -> None:
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
    


class TFBertSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs) -> None:
        ...
    
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool = ...) -> Tuple[tf.Tensor]:
        ...
    


class TFBertSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...
    


class TFBertAttention(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs) -> None:
        ...
    
    def prune_heads(self, heads):
        ...
    
    def call(self, input_tensor: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool = ...) -> Tuple[tf.Tensor]:
        ...
    


class TFBertIntermediate(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...
    


class TFBertOutput(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...
    


class TFBertLayer(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool = ...) -> Tuple[tf.Tensor]:
        ...
    


class TFBertEncoder(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool = ...) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        ...
    


class TFBertPooler(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...
    


class TFBertPredictionHeadTransform(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...
    


class TFBertLMPredictionHead(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, input_embeddings: tf.keras.layers.Layer, **kwargs) -> None:
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
    


class TFBertMLMHead(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, input_embeddings: tf.keras.layers.Layer, **kwargs) -> None:
        ...
    
    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        ...
    


class TFBertNSPHead(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs) -> None:
        ...
    
    def call(self, pooled_output: tf.Tensor) -> tf.Tensor:
        ...
    


@keras_serializable
class TFBertMainLayer(tf.keras.layers.Layer):
    config_class = BertConfig
    def __init__(self, config: BertConfig, add_pooling_layer: bool = ..., **kwargs) -> None:
        ...
    
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        ...
    
    def set_input_embeddings(self, value: tf.Variable): # -> None:
        ...
    
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: bool = ..., **kwargs) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        ...
    


class TFBertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BertConfig
    base_model_prefix = ...


@dataclass
class TFBertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.TFBertForPreTraining`.

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
    hidden_states: Optional[Union[Tuple[tf.Tensor], tf.Tensor]] = ...
    attentions: Optional[Union[Tuple[tf.Tensor], tf.Tensor]] = ...


BERT_START_DOCSTRING = ...
BERT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare Bert Model transformer outputting raw hidden-states without any specific head on top.", BERT_START_DOCSTRING)
class TFBertModel(TFBertPreTrainedModel):
    def __init__(self, config: BertConfig, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        ...
    
    def serving_output(self, output: TFBaseModelOutputWithPooling) -> TFBaseModelOutputWithPooling:
        ...
    


@add_start_docstrings("""
Bert Model with two heads on top as done during the pretraining:
    a `masked language modeling` head and a `next sentence prediction (classification)` head.
    """, BERT_START_DOCSTRING)
class TFBertForPreTraining(TFBertPreTrainedModel, TFBertPreTrainingLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: BertConfig, *inputs, **kwargs) -> None:
        ...
    
    def get_lm_head(self) -> tf.keras.layers.Layer:
        ...
    
    def get_prefix_bias_name(self) -> str:
        ...
    
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[Union[np.ndarray, tf.Tensor]] = ..., next_sentence_label: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFBertForPreTrainingOutput, Tuple[tf.Tensor]]:
        r"""
        labels (:obj:`torch.LongTensor` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        next_sentence_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Return:

        Examples::

            >>> import tensorflow as tf
            >>> from transformers import BertTokenizer, TFBertForPreTraining

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = TFBertForPreTraining.from_pretrained('bert-base-uncased')
            >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]  # Batch size 1
            >>> outputs = model(input_ids)
            >>> prediction_scores, seq_relationship_scores = outputs[:2]

        """
        ...
    
    def serving_output(self, output: TFBertForPreTrainingOutput) -> TFBertForPreTrainingOutput:
        ...
    


@add_start_docstrings("""Bert Model with a `language modeling` head on top. """, BERT_START_DOCSTRING)
class TFBertForMaskedLM(TFBertPreTrainedModel, TFMaskedLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: BertConfig, *inputs, **kwargs) -> None:
        ...
    
    def get_lm_head(self) -> tf.keras.layers.Layer:
        ...
    
    def get_prefix_bias_name(self) -> str:
        ...
    
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (:obj:`tf.Tensor` or :obj:`np.ndarray` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        ...
    
    def serving_output(self, output: TFMaskedLMOutput) -> TFMaskedLMOutput:
        ...
    


class TFBertLMHeadModel(TFBertPreTrainedModel, TFCausalLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: BertConfig, *inputs, **kwargs) -> None:
        ...
    
    def get_lm_head(self) -> tf.keras.layers.Layer:
        ...
    
    def get_prefix_bias_name(self) -> str:
        ...
    
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFCausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFCausalLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (:obj:`tf.Tensor` or :obj:`np.ndarray` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss. Indices should be in ``[0, ...,
            config.vocab_size - 1]``.
        """
        ...
    
    def serving_output(self, output: TFCausalLMOutput) -> TFCausalLMOutput:
        ...
    


@add_start_docstrings("""Bert Model with a `next sentence prediction (classification)` head on top. """, BERT_START_DOCSTRING)
class TFBertForNextSentencePrediction(TFBertPreTrainedModel, TFNextSentencePredictionLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: BertConfig, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFNextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., next_sentence_label: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFNextSentencePredictorOutput, Tuple[tf.Tensor]]:
        r"""
        Return:

        Examples::

            >>> import tensorflow as tf
            >>> from transformers import BertTokenizer, TFBertForNextSentencePrediction

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = TFBertForNextSentencePrediction.from_pretrained('bert-base-uncased')

            >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
            >>> encoding = tokenizer(prompt, next_sentence, return_tensors='tf')

            >>> logits = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])[0]
            >>> assert logits[0][0] < logits[0][1] # the next sentence was random
        """
        ...
    
    def serving_output(self, output: TFNextSentencePredictorOutput) -> TFNextSentencePredictorOutput:
        ...
    


@add_start_docstrings("""
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """, BERT_START_DOCSTRING)
class TFBertForSequenceClassification(TFBertPreTrainedModel, TFSequenceClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config: BertConfig, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
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
    Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, BERT_START_DOCSTRING)
class TFBertForMultipleChoice(TFBertPreTrainedModel, TFMultipleChoiceLoss):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config: BertConfig, *inputs, **kwargs) -> None:
        ...
    
    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        ...
    
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
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
    Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, BERT_START_DOCSTRING)
class TFBertForTokenClassification(TFBertPreTrainedModel, TFTokenClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config: BertConfig, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (:obj:`tf.Tensor` or :obj:`np.ndarray` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        ...
    
    def serving_output(self, output: TFTokenClassifierOutput) -> TFTokenClassifierOutput:
        ...
    


@add_start_docstrings("""
    Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, BERT_START_DOCSTRING)
class TFBertForQuestionAnswering(TFBertPreTrainedModel, TFQuestionAnsweringLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: BertConfig, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., start_positions: Optional[Union[np.ndarray, tf.Tensor]] = ..., end_positions: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
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
    


