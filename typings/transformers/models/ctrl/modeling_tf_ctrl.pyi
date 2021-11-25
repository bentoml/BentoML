

import tensorflow as tf

from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPast,
    TFCausalLMOutputWithPast,
    TFSequenceClassifierOutput,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras_serializable,
)
from .configuration_ctrl import CTRLConfig

""" TF 2.0 CTRL model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
TF_CTRL_PRETRAINED_MODEL_ARCHIVE_LIST = ...
def angle_defn(pos, i, d_model_size):
    ...

def positional_encoding(position, d_model_size):
    ...

def scaled_dot_product_attention(q, k, v, mask, attention_mask=..., head_mask=...): # -> tuple[Unknown, Unknown]:
    ...

class TFMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model_size, num_heads, output_attentions=..., **kwargs) -> None:
        ...
    
    def split_into_heads(self, x, batch_size):
        ...
    
    def call(self, v, k, q, mask, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=...): # -> tuple[Unknown, Unknown | tuple[None], Unknown] | tuple[Unknown, Unknown | tuple[None]]:
        ...
    


class TFPointWiseFeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self, d_model_size, dff, **kwargs) -> None:
        ...
    
    def call(self, inputs, trainable=...):
        ...
    


class TFEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model_size, num_heads, dff, rate=..., layer_norm_epsilon=..., output_attentions=..., **kwargs) -> None:
        ...
    
    def call(self, x, mask, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=...):
        ...
    


@keras_serializable
class TFCTRLMainLayer(tf.keras.layers.Layer):
    config_class = CTRLConfig
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def get_input_embeddings(self): # -> TFSharedEmbeddings:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    def call(self, input_ids=..., past=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    


class TFCTRLPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = CTRLConfig
    base_model_prefix = ...


CTRL_START_DOCSTRING = ...
CTRL_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare CTRL Model transformer outputting raw hidden-states without any specific head on top.", CTRL_START_DOCSTRING)
class TFCTRLModel(TFCTRLPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., past=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    
    def serving_output(self, output): # -> TFBaseModelOutputWithPast:
        ...
    


class TFCTRLLMHead(tf.keras.layers.Layer):
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
    


@add_start_docstrings("""
    The CTRL Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """, CTRL_START_DOCSTRING)
class TFCTRLLMHeadModel(TFCTRLPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def get_lm_head(self): # -> TFCTRLLMHead:
        ...
    
    def get_prefix_bias_name(self):
        ...
    
    def prepare_inputs_for_generation(self, inputs, past, **kwargs): # -> dict[str, Unknown]:
        ...
    
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., past=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs): # -> TFCausalLMOutputWithPast:
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss. Indices should be in ``[0, ...,
            config.vocab_size - 1]``.
        """
        ...
    
    def serving_output(self, output): # -> TFCausalLMOutputWithPast:
        ...
    


@add_start_docstrings("""
    The CTRL Model transformer with a sequence classification head on top (linear layer).

    :class:`~transformers.TFCTRLForSequenceClassification` uses the last token in order to do the classification, as
    other causal models (e.g. GPT-1, GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    :obj:`pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each
    row. If no :obj:`pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot
    guess the padding tokens when :obj:`inputs_embeds` are passed instead of :obj:`input_ids`, it does the same (take
    the last value in each row of the batch).
    """, CTRL_START_DOCSTRING)
class TFCTRLForSequenceClassification(TFCTRLPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def get_output_embeddings(self): # -> TFSharedEmbeddings:
        ...
    
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., past=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss. Indices should be in ``[0, ...,
            config.vocab_size - 1]``.
        """
        ...
    
    def serving_output(self, output: TFSequenceClassifierOutput) -> TFSequenceClassifierOutput:
        ...
    


