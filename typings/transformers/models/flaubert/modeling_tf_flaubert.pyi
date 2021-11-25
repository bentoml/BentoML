

from dataclasses import dataclass
from typing import Optional, Tuple

import tensorflow as tf

from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_tf_outputs import TFBaseModelOutput
from ...modeling_tf_utils import TFPreTrainedModel, keras_serializable
from ..xlm.modeling_tf_xlm import (
    TFXLMForMultipleChoice,
    TFXLMForQuestionAnsweringSimple,
    TFXLMForSequenceClassification,
    TFXLMForTokenClassification,
)
from .configuration_flaubert import FlaubertConfig

"""
 TF 2.0 Flaubert model.
"""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
TF_FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
FLAUBERT_START_DOCSTRING = ...
FLAUBERT_INPUTS_DOCSTRING = ...
def get_masks(slen, lengths, causal, padding_mask=...): # -> tuple[Unknown, Unknown]:
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    ...

class TFFlaubertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = FlaubertConfig
    base_model_prefix = ...
    @property
    def dummy_inputs(self): # -> dict[str, Unknown]:
        ...
    


@add_start_docstrings("The bare Flaubert Model transformer outputting raw hidden-states without any specific head on top.", FLAUBERT_START_DOCSTRING)
class TFFlaubertModel(TFFlaubertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., langs=..., token_type_ids=..., position_ids=..., lengths=..., cache=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    
    def serving_output(self, output): # -> TFBaseModelOutput:
        ...
    


class TFFlaubertMultiHeadAttention(tf.keras.layers.Layer):
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
    


class TFFlaubertTransformerFFN(tf.keras.layers.Layer):
    def __init__(self, in_dim, dim_hidden, out_dim, config, **kwargs) -> None:
        ...
    
    def call(self, input, training=...):
        ...
    


@keras_serializable
class TFFlaubertMainLayer(tf.keras.layers.Layer):
    config_class = FlaubertConfig
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
    


class TFFlaubertPredLayer(tf.keras.layers.Layer):
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
    


@dataclass
class TFFlaubertWithLMHeadModelOutput(ModelOutput):
    """
    Base class for :class:`~transformers.TFFlaubertWithLMHeadModel` outputs.

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


@add_start_docstrings("""
    The Flaubert Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """, FLAUBERT_START_DOCSTRING)
class TFFlaubertWithLMHeadModel(TFFlaubertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def get_lm_head(self): # -> TFFlaubertPredLayer:
        ...
    
    def get_prefix_bias_name(self):
        ...
    
    def prepare_inputs_for_generation(self, inputs, **kwargs): # -> dict[str, Unknown | None]:
        ...
    
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFFlaubertWithLMHeadModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., langs=..., token_type_ids=..., position_ids=..., lengths=..., cache=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs): # -> TFFlaubertWithLMHeadModelOutput:
        ...
    
    def serving_output(self, output): # -> TFFlaubertWithLMHeadModelOutput:
        ...
    


@add_start_docstrings("""
    Flaubert Model with a sequence classification/regression head on top (a linear layer on top of the pooled output)
    e.g. for GLUE tasks.
    """, FLAUBERT_START_DOCSTRING)
class TFFlaubertForSequenceClassification(TFXLMForSequenceClassification):
    config_class = FlaubertConfig
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    


@add_start_docstrings("""
    Flaubert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, FLAUBERT_START_DOCSTRING)
class TFFlaubertForQuestionAnsweringSimple(TFXLMForQuestionAnsweringSimple):
    config_class = FlaubertConfig
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    


@add_start_docstrings("""
    Flaubert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, FLAUBERT_START_DOCSTRING)
class TFFlaubertForTokenClassification(TFXLMForTokenClassification):
    config_class = FlaubertConfig
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    


@add_start_docstrings("""
    Flaubert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, FLAUBERT_START_DOCSTRING)
class TFFlaubertForMultipleChoice(TFXLMForMultipleChoice):
    config_class = FlaubertConfig
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    


