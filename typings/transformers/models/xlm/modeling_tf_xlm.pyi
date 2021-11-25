

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import tensorflow as tf

from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    keras_serializable,
)
from .configuration_xlm import XLMConfig

"""
 TF 2.0 XLM model.
"""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
TF_XLM_PRETRAINED_MODEL_ARCHIVE_LIST = ...
def create_sinusoidal_embeddings(n_pos, dim, out): # -> None:
    ...

def get_masks(slen, lengths, causal, padding_mask=...): # -> tuple[Unknown, Unknown]:
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    ...

class TFXLMMultiHeadAttention(tf.keras.layers.Layer):
    NEW_ID = ...
    def __init__(self, n_heads, dim, config, **kwargs) -> None:
        ...
    
    def prune_heads(self, heads):
        ...
    
    def call(self, input, mask, kv, cache, head_mask, output_attentions, training=...):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        ...
    


class TFXLMTransformerFFN(tf.keras.layers.Layer):
    def __init__(self, in_dim, dim_hidden, out_dim, config, **kwargs) -> None:
        ...
    
    def call(self, input, training=...):
        ...
    


@keras_serializable
class TFXLMMainLayer(tf.keras.layers.Layer):
    config_class = XLMConfig
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def build(self, input_shape): # -> None:
        ...
    
    def get_input_embeddings(self): # -> TFSharedEmbeddings:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    def call(self, input_ids=..., attention_mask=..., langs=..., token_type_ids=..., position_ids=..., lengths=..., cache=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    


class TFXLMPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = XLMConfig
    base_model_prefix = ...
    @property
    def dummy_inputs(self): # -> dict[str, Unknown]:
        ...
    


@dataclass
class TFXLMWithLMHeadModelOutput(ModelOutput):
    """
    Base class for :class:`~transformers.TFXLMWithLMHeadModel` outputs.

    Args:
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
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
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...


XLM_START_DOCSTRING = ...
XLM_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare XLM Model transformer outputting raw hidden-states without any specific head on top.", XLM_START_DOCSTRING)
class TFXLMModel(TFXLMPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., langs=..., token_type_ids=..., position_ids=..., lengths=..., cache=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    
    def serving_output(self, output): # -> TFBaseModelOutput:
        ...
    


class TFXLMPredLayer(tf.keras.layers.Layer):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """
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
    The XLM Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """, XLM_START_DOCSTRING)
class TFXLMWithLMHeadModel(TFXLMPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def get_lm_head(self): # -> TFXLMPredLayer:
        ...
    
    def get_prefix_bias_name(self):
        ...
    
    def prepare_inputs_for_generation(self, inputs, **kwargs): # -> dict[str, Unknown | None]:
        ...
    
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFXLMWithLMHeadModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., langs=..., token_type_ids=..., position_ids=..., lengths=..., cache=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs): # -> TFXLMWithLMHeadModelOutput:
        ...
    
    def serving_output(self, output): # -> TFXLMWithLMHeadModelOutput:
        ...
    


@add_start_docstrings("""
    XLM Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g.
    for GLUE tasks.
    """, XLM_START_DOCSTRING)
class TFXLMForSequenceClassification(TFXLMPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., langs=..., token_type_ids=..., position_ids=..., lengths=..., cache=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs): # -> TFSequenceClassifierOutput:
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
    XLM Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, XLM_START_DOCSTRING)
class TFXLMForMultipleChoice(TFXLMPreTrainedModel, TFMultipleChoiceLoss):
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
    
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., langs=..., token_type_ids=..., position_ids=..., lengths=..., cache=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs):
        ...
    
    @tf.function(input_signature=[{ "input_ids": tf.TensorSpec((None, None, None), tf.int32, name="input_ids"),"attention_mask": tf.TensorSpec((None, None, None), tf.int32, name="attention_mask"),"token_type_ids": tf.TensorSpec((None, None, None), tf.int32, name="token_type_ids") }])
    def serving(self, inputs: Dict[str, tf.Tensor]): # -> TFMultipleChoiceModelOutput:
        ...
    
    def serving_output(self, output: TFMultipleChoiceModelOutput) -> TFMultipleChoiceModelOutput:
        ...
    


@add_start_docstrings("""
    XLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, XLM_START_DOCSTRING)
class TFXLMForTokenClassification(TFXLMPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., langs=..., token_type_ids=..., position_ids=..., lengths=..., cache=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs): # -> TFTokenClassifierOutput:
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        ...
    
    def serving_output(self, output: TFTokenClassifierOutput) -> TFTokenClassifierOutput:
        ...
    


@add_start_docstrings("""
    XLM Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layer
    on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, XLM_START_DOCSTRING)
class TFXLMForQuestionAnsweringSimple(TFXLMPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., langs=..., token_type_ids=..., position_ids=..., lengths=..., cache=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., start_positions=..., end_positions=..., training=..., **kwargs): # -> TFQuestionAnsweringModelOutput:
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
    


