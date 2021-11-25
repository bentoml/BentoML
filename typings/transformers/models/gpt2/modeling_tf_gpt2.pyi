

from dataclasses import dataclass
from typing import List, Optional, Tuple

import tensorflow as tf

from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPast,
    TFCausalLMOutputWithPast,
    TFSequenceClassifierOutputWithPast,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras_serializable,
)
from .configuration_gpt2 import GPT2Config

""" TF 2.0 OpenAI GPT-2 model. """
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class TFAttention(tf.keras.layers.Layer):
    def __init__(self, nx, n_ctx, config, scale=..., **kwargs) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    @staticmethod
    def causal_attention_mask(nd, ns, dtype):
        """
        1's in the lower triangle, counting from the lower right corner. Same as tf.matrix_band_part(tf.ones([nd, ns]),
        -1, ns-nd), but doesn't produce garbage on TPUs.
        """
        ...
    
    def merge_heads(self, x):
        ...
    
    def split_heads(self, x):
        ...
    
    def call(self, x, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=...): # -> list[Unknown | tuple[None]]:
        ...
    


class TFMLP(tf.keras.layers.Layer):
    def __init__(self, n_state, config, **kwargs) -> None:
        ...
    
    def call(self, x, training=...):
        ...
    


class TFBlock(tf.keras.layers.Layer):
    def __init__(self, n_ctx, config, scale=..., **kwargs) -> None:
        ...
    
    def call(self, x, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=...):
        ...
    


@keras_serializable
class TFGPT2MainLayer(tf.keras.layers.Layer):
    config_class = GPT2Config
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def build(self, input_shape): # -> None:
        ...
    
    def get_input_embeddings(self): # -> TFSharedEmbeddings:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    def call(self, input_ids=..., past=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    


class TFGPT2PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GPT2Config
    base_model_prefix = ...
    _keys_to_ignore_on_load_unexpected = ...
    @tf.function(input_signature=[{ "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),"attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask") }])
    def serving(self, inputs):
        ...
    


@dataclass
class TFGPT2DoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.

    Args:
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past_key_values (:obj:`List[tf.Tensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`tf.Tensor` of length :obj:`config.n_layers`, with each tensor of shape :obj:`(2, batch_size,
            num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
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
    past_key_values: Optional[List[tf.Tensor]] = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...


GPT2_START_DOCSTRING = ...
GPT2_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.", GPT2_START_DOCSTRING)
class TFGPT2Model(TFGPT2PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., past=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    
    def serving_output(self, output): # -> TFBaseModelOutputWithPast:
        ...
    


@add_start_docstrings("""
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """, GPT2_START_DOCSTRING)
class TFGPT2LMHeadModel(TFGPT2PreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def get_output_embeddings(self):
        ...
    
    def set_output_embeddings(self, value): # -> None:
        ...
    
    def prepare_inputs_for_generation(self, inputs, past, **kwargs): # -> dict[str, Unknown]:
        ...
    
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
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
    The GPT2 Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
    RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the
    input embeddings, the classification head takes as input the input of a specified classification token index in the
    input sequence).
    """, GPT2_START_DOCSTRING)
class TFGPT2DoubleHeadsModel(TFGPT2PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFGPT2DoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., past=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., mc_token_ids=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs): # -> TFGPT2DoubleHeadsModelOutput:
        r"""
        mc_token_ids (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, num_choices)`, `optional`, default to index of the last token of the input):
            Index of the classification token in each input sequence. Selected in the range ``[0, input_ids.size(-1) -
            1[``.

        Return:

        Examples::

            >>> import tensorflow as tf
            >>> from transformers import GPT2Tokenizer, TFGPT2DoubleHeadsModel

            >>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            >>> model = TFGPT2DoubleHeadsModel.from_pretrained('gpt2')

            >>> # Add a [CLS] to the vocabulary (we should train it also!)
            >>> num_added_tokens = tokenizer.add_special_tokens({'cls_token': '[CLS]'})

            >>> embedding_layer = model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size

            >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
            >>> encoded_choices = [tokenizer.encode(s) for s in choices]
            >>> cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

            >>> input_ids = tf.constant(encoded_choices)[None, :]  # Batch size: 1, number of choices: 2
            >>> mc_token_ids = tf.constant([cls_token_location])  # Batch size: 1

            >>> outputs = model(input_ids, mc_token_ids=mc_token_ids)
            >>> lm_prediction_scores, mc_prediction_scores = outputs[:2]

        """
        ...
    
    @tf.function(input_signature=[{ "input_ids": tf.TensorSpec((None, None, None), tf.int32, name="input_ids"),"attention_mask": tf.TensorSpec((None, None, None), tf.int32, name="attention_mask"),"mc_token_ids": tf.TensorSpec((None, None), tf.int32, name="mc_token_ids") }])
    def serving(self, inputs): # -> TFGPT2DoubleHeadsModelOutput:
        ...
    
    def serving_output(self, output): # -> TFGPT2DoubleHeadsModelOutput:
        ...
    


@add_start_docstrings("""
    The GPT2 Model transformer with a sequence classification head on top (linear layer).

    :class:`~transformers.TFGPT2ForSequenceClassification` uses the last token in order to do the classification, as
    other causal models (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    :obj:`pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each
    row. If no :obj:`pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot
    guess the padding tokens when :obj:`inputs_embeds` are passed instead of :obj:`input_ids`, it does the same (take
    the last value in each row of the batch).
    """, GPT2_START_DOCSTRING)
class TFGPT2ForSequenceClassification(TFGPT2PreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="microsoft/DialogRPT-updown", output_type=TFSequenceClassifierOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., past=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss. Indices should be in ``[0, ...,
            config.vocab_size - 1]``.
        """
        ...
    
    def serving_output(self, output): # -> TFSequenceClassifierOutputWithPast:
        ...
    


