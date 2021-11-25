

from typing import Tuple

import tensorflow as tf

from ...file_utils import (
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
    keras_serializable,
)
from .configuration_t5 import T5Config

""" TF 2.0 T5 model. """
logger = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class TFT5LayerNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=..., **kwargs) -> None:
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        ...
    
    def build(self, input_shape): # -> None:
        """Build shared word embedding layer"""
        ...
    
    def call(self, hidden_states):
        ...
    


class TFT5DenseReluDense(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states, training=...):
        ...
    


class TFT5GatedGeluDense(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states, training=...):
        ...
    


class TFT5LayerFF(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states, training=...):
        ...
    


class TFT5Attention(tf.keras.layers.Layer):
    NEW_ID = ...
    def __init__(self, config, has_relative_attention_bias=..., **kwargs) -> None:
        ...
    
    def build(self, input_shape):
        ...
    
    def prune_heads(self, heads):
        ...
    
    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        ...
    
    def call(self, hidden_states, mask=..., key_value_states=..., position_bias=..., past_key_value=..., layer_head_mask=..., query_length=..., use_cache=..., training=..., output_attentions=...):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        ...
    


class TFT5LayerSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, has_relative_attention_bias=..., **kwargs) -> None:
        ...
    
    def call(self, hidden_states, attention_mask=..., position_bias=..., layer_head_mask=..., past_key_value=..., use_cache=..., output_attentions=..., training=...):
        ...
    


class TFT5LayerCrossAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states, key_value_states, attention_mask=..., position_bias=..., layer_head_mask=..., past_key_value=..., query_length=..., use_cache=..., output_attentions=..., training=...):
        ...
    


class TFT5Block(tf.keras.layers.Layer):
    def __init__(self, config, has_relative_attention_bias=..., **kwargs) -> None:
        ...
    
    def call(self, hidden_states, attention_mask=..., position_bias=..., encoder_hidden_states=..., encoder_attention_mask=..., encoder_decoder_position_bias=..., layer_head_mask=..., encoder_layer_head_mask=..., past_key_value=..., use_cache=..., output_attentions=..., training=...):
        ...
    


@keras_serializable
class TFT5MainLayer(tf.keras.layers.Layer):
    config_class = T5Config
    def __init__(self, config, embed_tokens=..., **kwargs) -> None:
        ...
    
    def call(self, input_ids=..., attention_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., inputs_embeds=..., head_mask=..., encoder_head_mask=..., past_key_values=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs) -> Tuple:
        ...
    


class TFT5PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = T5Config
    base_model_prefix = ...
    _keys_to_ignore_on_load_unexpected = ...
    @property
    def dummy_inputs(self): # -> dict[str, Unknown]:
        ...
    
    @tf.function(input_signature=[{ "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),"attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),"decoder_input_ids": tf.TensorSpec((None, None), tf.int32, name="decoder_input_ids"),"decoder_attention_mask": tf.TensorSpec((None, None), tf.int32, name="decoder_attention_mask") }])
    def serving(self, inputs):
        ...
    
    def get_input_embeddings(self):
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    


T5_START_DOCSTRING = ...
T5_INPUTS_DOCSTRING = ...
T5_ENCODER_INPUTS_DOCSTRING = ...
_HEAD_MASK_WARNING_MSG = ...
@add_start_docstrings("The bare T5 Model transformer outputting raw hidden-states" "without any specific head on top.", T5_START_DOCSTRING)
class TFT5Model(TFT5PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def get_encoder(self): # -> TFT5MainLayer:
        ...
    
    def get_decoder(self): # -> TFT5MainLayer:
        ...
    
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., decoder_input_ids=..., decoder_attention_mask=..., head_mask=..., decoder_head_mask=..., encoder_outputs=..., past_key_values=..., inputs_embeds=..., decoder_inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        r"""
        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, TFT5Model

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = TFT5Model.from_pretrained('t5-small')

            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="tf").input_ids  # Batch size 1
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="tf").input_ids  # Batch size 1
            >>> outputs = model(input_ids, decoder_input_ids=decoder_input_ids)


        """
        ...
    
    def serving_output(self, output): # -> TFSeq2SeqModelOutput:
        ...
    


@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
class TFT5ForConditionalGeneration(TFT5PreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def get_output_embeddings(self):
        ...
    
    def set_output_embeddings(self, value): # -> None:
        ...
    
    def get_encoder(self): # -> TFT5MainLayer:
        ...
    
    def get_decoder(self): # -> TFT5MainLayer:
        ...
    
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., decoder_input_ids=..., decoder_attention_mask=..., head_mask=..., decoder_head_mask=..., encoder_outputs=..., past_key_values=..., inputs_embeds=..., decoder_inputs_embeds=..., labels=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss. Indices should be in ``[0, ...,
            config.vocab_size - 1]``.

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, TFT5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = TFT5ForConditionalGeneration.from_pretrained('t5-small')

            >>> inputs = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='tf').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='tf').input_ids
            >>> outputs = model(inputs, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> inputs = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="tf").input_ids  # Batch size 1

            >>> result = model.generate(inputs)

        """
        ...
    
    def serving_output(self, output): # -> TFSeq2SeqLMOutput:
        ...
    
    def prepare_inputs_for_generation(self, inputs, past, attention_mask, use_cache=..., **kwargs): # -> dict[str, Unknown | tuple[Unknown, ...] | None]:
        ...
    
    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        ...
    


@add_start_docstrings("The bare T5 Model transformer outputting encoder's raw hidden-states" "without any specific head on top.", T5_START_DOCSTRING)
class TFT5EncoderModel(TFT5PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def get_encoder(self): # -> TFT5MainLayer:
        ...
    
    @add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids, attention_mask=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs): # -> TFBaseModelOutput:
        r"""
        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, TFT5Model

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = TFT5EncoderModel.from_pretrained('t5-small')

            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="tf").input_ids  # Batch size 1
            >>> outputs = model(input_ids)


        """
        ...
    
    def serving_output(self, output): # -> TFBaseModelOutput:
        ...
    


