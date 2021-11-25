

from dataclasses import dataclass
from typing import Optional, Tuple

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
    TFCausalLMOutput,
    TFSequenceClassifierOutput,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras_serializable,
)
from .configuration_openai import OpenAIGPTConfig

""" TF 2.0 OpenAI GPT model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class TFAttention(tf.keras.layers.Layer):
    def __init__(self, nx, n_ctx, config, scale=..., **kwargs) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    @staticmethod
    def causal_attention_mask(nd, ns):
        """
        1's in the lower triangle, counting from the lower right corner. Same as tf.matrix_band_part(tf.ones([nd, ns]),
        -1, ns-nd), but doesn't produce garbage on TPUs.
        """
        ...
    
    def merge_heads(self, x):
        ...
    
    def split_heads(self, x):
        ...
    
    def call(self, x, attention_mask, head_mask, output_attentions, training=...): # -> list[Unknown]:
        ...
    


class TFMLP(tf.keras.layers.Layer):
    def __init__(self, n_state, config, **kwargs) -> None:
        ...
    
    def call(self, x, training=...):
        ...
    


class TFBlock(tf.keras.layers.Layer):
    def __init__(self, n_ctx, config, scale=..., **kwargs) -> None:
        ...
    
    def call(self, x, attention_mask, head_mask, output_attentions, training=...):
        ...
    


@keras_serializable
class TFOpenAIGPTMainLayer(tf.keras.layers.Layer):
    config_class = OpenAIGPTConfig
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def build(self, input_shape): # -> None:
        ...
    
    def get_input_embeddings(self): # -> TFSharedEmbeddings:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    


class TFOpenAIGPTPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = OpenAIGPTConfig
    base_model_prefix = ...
    @tf.function(input_signature=[{ "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),"attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask") }])
    def serving(self, inputs):
        ...
    


@dataclass
class TFOpenAIGPTDoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.

    Args:
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
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
    logits: tf.Tensor = ...
    mc_logits: tf.Tensor = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...


OPENAI_GPT_START_DOCSTRING = ...
OPENAI_GPT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare OpenAI GPT transformer model outputting raw hidden-states without any specific head on top.", OPENAI_GPT_START_DOCSTRING)
class TFOpenAIGPTModel(TFOpenAIGPTPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    
    def serving_output(self, output): # -> TFBaseModelOutput:
        ...
    


@add_start_docstrings("""
    OpenAI GPT Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """, OPENAI_GPT_START_DOCSTRING)
class TFOpenAIGPTLMHeadModel(TFOpenAIGPTPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def get_output_embeddings(self):
        ...
    
    def set_output_embeddings(self, value): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFCausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs): # -> TFCausalLMOutput:
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss. Indices should be in ``[0, ...,
            config.vocab_size - 1]``.
        """
        ...
    
    def serving_output(self, output: TFCausalLMOutput) -> TFCausalLMOutput:
        ...
    


@add_start_docstrings("""
    OpenAI GPT Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
    RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the
    input embeddings, the classification head takes as input the input of a specified classification token index in the
    input sequence).
    """, OPENAI_GPT_START_DOCSTRING)
class TFOpenAIGPTDoubleHeadsModel(TFOpenAIGPTPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFOpenAIGPTDoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., mc_token_ids=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs): # -> TFOpenAIGPTDoubleHeadsModelOutput:
        r"""
        mc_token_ids (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, num_choices)`, `optional`, default to index of the last token of the input):
            Index of the classification token in each input sequence. Selected in the range ``[0, input_ids.size(-1) -
            1]``.

        Return:

        Examples::

            >>> import tensorflow as tf
            >>> from transformers import OpenAIGPTTokenizer, TFOpenAIGPTDoubleHeadsModel

            >>> tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
            >>> model = TFOpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt')

            >>> # Add a [CLS] to the vocabulary (we should train it also!)
            >>> tokenizer.add_special_tokens({'cls_token': '[CLS]'})
            >>> model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size
            >>> print(tokenizer.cls_token_id, len(tokenizer))  # The newly token the last token of the vocabulary

            >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
            >>> encoding = tokenizer(choices, return_tensors="tf")
            >>> inputs = {k: tf.expand_dims(v, 0) for k, v in encoding.items()}
            >>> inputs["mc_token_ids"]= tf.constant([inputs["input_ids"].shape[-1] - 1, inputs["input_ids"].shape[-1] - 1])[None, :]  # Batch size 1
            >>> outputs = model(inputs)
            >>> lm_prediction_scores, mc_prediction_scores = outputs[:2]
        """
        ...
    
    @tf.function(input_signature=[{ "input_ids": tf.TensorSpec((None, None, None), tf.int32, name="input_ids"),"attention_mask": tf.TensorSpec((None, None, None), tf.int32, name="attention_mask"),"mc_token_ids": tf.TensorSpec((None, None), tf.int32, name="token_type_ids") }])
    def serving(self, inputs): # -> TFOpenAIGPTDoubleHeadsModelOutput:
        ...
    
    def serving_output(self, output): # -> TFOpenAIGPTDoubleHeadsModelOutput:
        ...
    


@add_start_docstrings("""
    The OpenAI GPT Model transformer with a sequence classification head on top (linear layer).

    :class:`~transformers.TFOpenAIGPTForSequenceClassification` uses the last token in order to do the classification,
    as other causal models (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    :obj:`pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each
    row. If no :obj:`pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot
    guess the padding tokens when :obj:`inputs_embeds` are passed instead of :obj:`input_ids`, it does the same (take
    the last value in each row of the batch).
    """, OPENAI_GPT_START_DOCSTRING)
class TFOpenAIGPTForSequenceClassification(TFOpenAIGPTPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss. Indices should be in ``[0, ...,
            config.vocab_size - 1]``.
        """
        ...
    
    def serving_output(self, output: TFSequenceClassifierOutput) -> TFSequenceClassifierOutput:
        ...
    


