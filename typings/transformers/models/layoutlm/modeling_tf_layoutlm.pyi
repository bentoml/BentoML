

from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFMaskedLMOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    keras_serializable,
)
from .configuration_layoutlm import LayoutLMConfig

""" TF 2.0 LayoutLM model. """
logger = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
TF_LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class TFLayoutLMEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config: LayoutLMConfig, **kwargs) -> None:
        ...
    
    def build(self, input_shape: tf.TensorShape): # -> None:
        ...
    
    def call(self, input_ids: tf.Tensor = ..., bbox: tf.Tensor = ..., position_ids: tf.Tensor = ..., token_type_ids: tf.Tensor = ..., inputs_embeds: tf.Tensor = ..., training: bool = ...) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (:obj:`tf.Tensor`): output embedding tensor.
        """
        ...
    


class TFLayoutLMSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs) -> None:
        ...
    
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool = ...) -> Tuple[tf.Tensor]:
        ...
    


class TFLayoutLMSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...
    


class TFLayoutLMAttention(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs) -> None:
        ...
    
    def prune_heads(self, heads):
        ...
    
    def call(self, input_tensor: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool = ...) -> Tuple[tf.Tensor]:
        ...
    


class TFLayoutLMIntermediate(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...
    


class TFLayoutLMOutput(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...
    


class TFLayoutLMLayer(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool = ...) -> Tuple[tf.Tensor]:
        ...
    


class TFLayoutLMEncoder(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool = ...) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        ...
    


class TFLayoutLMPooler(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...
    


class TFLayoutLMPredictionHeadTransform(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...
    


class TFLayoutLMLMPredictionHead(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, input_embeddings: tf.keras.layers.Layer, **kwargs) -> None:
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
    


class TFLayoutLMMLMHead(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, input_embeddings: tf.keras.layers.Layer, **kwargs) -> None:
        ...
    
    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        ...
    


@keras_serializable
class TFLayoutLMMainLayer(tf.keras.layers.Layer):
    config_class = LayoutLMConfig
    def __init__(self, config: LayoutLMConfig, add_pooling_layer: bool = ..., **kwargs) -> None:
        ...
    
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        ...
    
    def set_input_embeddings(self, value: tf.Variable): # -> None:
        ...
    
    def call(self, input_ids: Optional[TFModelInputType] = ..., bbox: Optional[Union[np.ndarray, tf.Tensor]] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: bool = ..., **kwargs) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        ...
    


class TFLayoutLMPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = LayoutLMConfig
    base_model_prefix = ...


LAYOUTLM_START_DOCSTRING = ...
LAYOUTLM_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare LayoutLM Model transformer outputting raw hidden-states without any specific head on top.", LAYOUTLM_START_DOCSTRING)
class TFLayoutLMModel(TFLayoutLMPreTrainedModel):
    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., bbox: Optional[Union[np.ndarray, tf.Tensor]] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples::

            >>> from transformers import LayoutLMTokenizer, TFLayoutLMModel
            >>> import tensorflow as tf

            >>> tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
            >>> model = TFLayoutLMModel.from_pretrained('microsoft/layoutlm-base-uncased')

            >>> words = ["Hello", "world"]
            >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

            >>> token_boxes = []
            >>> for word, box in zip(words, normalized_word_boxes):
            ...     word_tokens = tokenizer.tokenize(word)
            ...     token_boxes.extend([box] * len(word_tokens))
            >>> # add bounding boxes of cls + sep tokens
            >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

            >>> encoding = tokenizer(' '.join(words), return_tensors="tf")
            >>> input_ids = encoding["input_ids"]
            >>> attention_mask = encoding["attention_mask"]
            >>> token_type_ids = encoding["token_type_ids"]
            >>> bbox = tf.convert_to_tensor([token_boxes])

            >>> outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)

            >>> last_hidden_states = outputs.last_hidden_state
        """
        ...
    
    def serving_output(self, output: TFBaseModelOutputWithPooling) -> TFBaseModelOutputWithPooling:
        ...
    


@add_start_docstrings("""LayoutLM Model with a `language modeling` head on top. """, LAYOUTLM_START_DOCSTRING)
class TFLayoutLMForMaskedLM(TFLayoutLMPreTrainedModel, TFMaskedLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs) -> None:
        ...
    
    def get_lm_head(self) -> tf.keras.layers.Layer:
        ...
    
    def get_prefix_bias_name(self) -> str:
        ...
    
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., bbox: Optional[Union[np.ndarray, tf.Tensor]] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (:obj:`tf.Tensor` or :obj:`np.ndarray` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import LayoutLMTokenizer, TFLayoutLMForMaskedLM
            >>> import tensorflow as tf

            >>> tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
            >>> model = TFLayoutLMForMaskedLM.from_pretrained('microsoft/layoutlm-base-uncased')

            >>> words = ["Hello", "[MASK]"]
            >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

            >>> token_boxes = []
            >>> for word, box in zip(words, normalized_word_boxes):
            ...     word_tokens = tokenizer.tokenize(word)
            ...     token_boxes.extend([box] * len(word_tokens))
            >>> # add bounding boxes of cls + sep tokens
            >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

            >>> encoding = tokenizer(' '.join(words), return_tensors="tf")
            >>> input_ids = encoding["input_ids"]
            >>> attention_mask = encoding["attention_mask"]
            >>> token_type_ids = encoding["token_type_ids"]
            >>> bbox = tf.convert_to_tensor([token_boxes])

            >>> labels = tokenizer("Hello world", return_tensors="tf")["input_ids"]

            >>> outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
            ...                 labels=labels)

            >>> loss = outputs.loss
        """
        ...
    
    def serving_output(self, output: TFMaskedLMOutput) -> TFMaskedLMOutput:
        ...
    


@add_start_docstrings("""
    LayoutLM Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """, LAYOUTLM_START_DOCSTRING)
class TFLayoutLMForSequenceClassification(TFLayoutLMPreTrainedModel, TFSequenceClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., bbox: Optional[Union[np.ndarray, tf.Tensor]] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (:obj:`tf.Tensor` or :obj:`np.ndarray` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples::

            >>> from transformers import LayoutLMTokenizer, TFLayoutLMForSequenceClassification
            >>> import tensorflow as tf

            >>> tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
            >>> model = TFLayoutLMForSequenceClassification.from_pretrained('microsoft/layoutlm-base-uncased')

            >>> words = ["Hello", "world"]
            >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

            >>> token_boxes = []
            >>> for word, box in zip(words, normalized_word_boxes):
            ...     word_tokens = tokenizer.tokenize(word)
            ...     token_boxes.extend([box] * len(word_tokens))
            >>> # add bounding boxes of cls + sep tokens
            >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

            >>> encoding = tokenizer(' '.join(words), return_tensors="tf")
            >>> input_ids = encoding["input_ids"]
            >>> attention_mask = encoding["attention_mask"]
            >>> token_type_ids = encoding["token_type_ids"]
            >>> bbox = tf.convert_to_tensor([token_boxes])
            >>> sequence_label = tf.convert_to_tensor([1])

            >>> outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
            ...                 labels=sequence_label)

            >>> loss = outputs.loss
            >>> logits = outputs.logits
        """
        ...
    
    def serving_output(self, output: TFSequenceClassifierOutput) -> TFSequenceClassifierOutput:
        ...
    


@add_start_docstrings("""
    LayoutLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, LAYOUTLM_START_DOCSTRING)
class TFLayoutLMForTokenClassification(TFLayoutLMPreTrainedModel, TFTokenClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: Optional[TFModelInputType] = ..., bbox: Optional[Union[np.ndarray, tf.Tensor]] = ..., attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., position_ids: Optional[Union[np.ndarray, tf.Tensor]] = ..., head_mask: Optional[Union[np.ndarray, tf.Tensor]] = ..., inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[Union[np.ndarray, tf.Tensor]] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (:obj:`tf.Tensor` or :obj:`np.ndarray` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.

        Returns:

        Examples::

            >>> from transformers import LayoutLMTokenizer, TFLayoutLMForTokenClassification
            >>> import torch

            >>> tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
            >>> model = TFLayoutLMForTokenClassification.from_pretrained('microsoft/layoutlm-base-uncased')

            >>> words = ["Hello", "world"]
            >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

            >>> token_boxes = []
            >>> for word, box in zip(words, normalized_word_boxes):
            ...     word_tokens = tokenizer.tokenize(word)
            ...     token_boxes.extend([box] * len(word_tokens))
            >>> # add bounding boxes of cls + sep tokens
            >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

            >>> encoding = tokenizer(' '.join(words), return_tensors="tf")
            >>> input_ids = encoding["input_ids"]
            >>> attention_mask = encoding["attention_mask"]
            >>> token_type_ids = encoding["token_type_ids"]
            >>> bbox = tf.convert_to_tensor([token_boxes])
            >>> token_labels = tf.convert_to_tensor([1,1,0,0])

            >>> outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
            ...                 labels=token_labels)

            >>> loss = outputs.loss
            >>> logits = outputs.logits
        """
        ...
    
    def serving_output(self, output: TFTokenClassifierOutput) -> TFTokenClassifierOutput:
        ...
    


