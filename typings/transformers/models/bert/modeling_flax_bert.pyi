

from typing import Callable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxlib.xla_extension as jax_xla
import numpy as np
from flax.core.frozen_dict import FrozenDict

from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_flax_utils import FlaxPreTrainedModel
from .configuration_bert import BertConfig

logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
@flax.struct.dataclass
class FlaxBertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.BertForPreTraining`.

    Args:
        prediction_logits (:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (:obj:`tuple(jax_xla.DeviceArray)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`jax_xla.DeviceArray` (one for the output of the embeddings + one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(jax_xla.DeviceArray)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`jax_xla.DeviceArray` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    prediction_logits: jax_xla.DeviceArray = ...
    seq_relationship_logits: jax_xla.DeviceArray = ...
    hidden_states: Optional[Tuple[jax_xla.DeviceArray]] = ...
    attentions: Optional[Tuple[jax_xla.DeviceArray]] = ...


BERT_START_DOCSTRING = ...
BERT_INPUTS_DOCSTRING = ...
class FlaxBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = ...):
        ...
    


class FlaxBertSelfAttention(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic=..., output_attentions: bool = ...): # -> tuple[Unknown, Unknown] | tuple[Unknown]:
        ...
    


class FlaxBertSelfOutput(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, input_tensor, deterministic: bool = ...):
        ...
    


class FlaxBertAttention(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic=..., output_attentions: bool = ...): # -> tuple[Unknown, Unknown] | tuple[Unknown]:
        ...
    


class FlaxBertIntermediate(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states):
        ...
    


class FlaxBertOutput(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_output, deterministic: bool = ...):
        ...
    


class FlaxBertLayer(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic: bool = ..., output_attentions: bool = ...): # -> tuple[Unknown, Unknown] | tuple[Unknown]:
        ...
    


class FlaxBertLayerCollection(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Unknown, ...] | FlaxBaseModelOutput:
        ...
    


class FlaxBertEncoder(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Unknown, ...] | FlaxBaseModelOutput:
        ...
    


class FlaxBertPooler(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states):
        ...
    


class FlaxBertPredictionHeadTransform(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states):
        ...
    


class FlaxBertLMPredictionHead(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    bias_init: Callable[..., np.ndarray] = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, shared_embedding=...):
        ...
    


class FlaxBertOnlyMLMHead(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, shared_embedding=...):
        ...
    


class FlaxBertOnlyNSPHead(nn.Module):
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, pooled_output):
        ...
    


class FlaxBertPreTrainingHeads(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, pooled_output, shared_embedding=...): # -> tuple[Unknown, Unknown]:
        ...
    


class FlaxBertPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BertConfig
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(self, config: BertConfig, input_shape: Tuple = ..., seed: int = ..., dtype: jnp.dtype = ..., **kwargs) -> None:
        ...
    
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        ...
    
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(self, input_ids, attention_mask=..., token_type_ids=..., position_ids=..., params: dict = ..., dropout_rng: jax.random.PRNGKey = ..., train: bool = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...):
        ...
    


class FlaxBertModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    add_pooling_layer: bool = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown | Any, ...] | Any | FlaxBaseModelOutputWithPooling:
        ...
    


@add_start_docstrings("The bare Bert Model transformer outputting raw hidden-states without any specific head on top.", BERT_START_DOCSTRING)
class FlaxBertModel(FlaxBertPreTrainedModel):
    module_class = ...


class FlaxBertForPreTrainingModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxBertForPreTrainingOutput:
        ...
    


@add_start_docstrings("""
    Bert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """, BERT_START_DOCSTRING)
class FlaxBertForPreTraining(FlaxBertPreTrainedModel):
    module_class = ...


FLAX_BERT_FOR_PRETRAINING_DOCSTRING = ...
class FlaxBertForMaskedLMModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxMaskedLMOutput:
        ...
    


@add_start_docstrings("""Bert Model with a `language modeling` head on top. """, BERT_START_DOCSTRING)
class FlaxBertForMaskedLM(FlaxBertPreTrainedModel):
    module_class = ...


class FlaxBertForNextSentencePredictionModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxNextSentencePredictorOutput:
        ...
    


@add_start_docstrings("""Bert Model with a `next sentence prediction (classification)` head on top. """, BERT_START_DOCSTRING)
class FlaxBertForNextSentencePrediction(FlaxBertPreTrainedModel):
    module_class = ...


FLAX_BERT_FOR_NEXT_SENT_PRED_DOCSTRING = ...
class FlaxBertForSequenceClassificationModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxSequenceClassifierOutput:
        ...
    


@add_start_docstrings("""
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """, BERT_START_DOCSTRING)
class FlaxBertForSequenceClassification(FlaxBertPreTrainedModel):
    module_class = ...


class FlaxBertForMultipleChoiceModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxMultipleChoiceModelOutput:
        ...
    


@add_start_docstrings("""
    Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, BERT_START_DOCSTRING)
class FlaxBertForMultipleChoice(FlaxBertPreTrainedModel):
    module_class = ...


class FlaxBertForTokenClassificationModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxTokenClassifierOutput:
        ...
    


@add_start_docstrings("""
    Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, BERT_START_DOCSTRING)
class FlaxBertForTokenClassification(FlaxBertPreTrainedModel):
    module_class = ...


class FlaxBertForQuestionAnsweringModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxQuestionAnsweringModelOutput:
        ...
    


@add_start_docstrings("""
    Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, BERT_START_DOCSTRING)
class FlaxBertForQuestionAnswering(FlaxBertPreTrainedModel):
    module_class = ...


