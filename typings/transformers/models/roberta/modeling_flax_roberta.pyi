

from typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from jax.random import PRNGKey

from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_flax_utils import FlaxPreTrainedModel
from .configuration_roberta import RobertaConfig

logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        input_ids: jnp.ndarray
        padding_idx: int

    Returns: jnp.ndarray
    """
    ...

ROBERTA_START_DOCSTRING = ...
ROBERTA_INPUTS_DOCSTRING = ...
class FlaxRobertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    config: RobertaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = ...):
        ...
    


class FlaxRobertaSelfAttention(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic=..., output_attentions: bool = ...): # -> tuple[Unknown, Unknown] | tuple[Unknown]:
        ...
    


class FlaxRobertaSelfOutput(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, input_tensor, deterministic: bool = ...):
        ...
    


class FlaxRobertaAttention(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic=..., output_attentions: bool = ...): # -> tuple[Unknown, Unknown] | tuple[Unknown]:
        ...
    


class FlaxRobertaIntermediate(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states):
        ...
    


class FlaxRobertaOutput(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_output, deterministic: bool = ...):
        ...
    


class FlaxRobertaLayer(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic: bool = ..., output_attentions: bool = ...): # -> tuple[Unknown, Unknown] | tuple[Unknown]:
        ...
    


class FlaxRobertaLayerCollection(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Unknown, ...] | FlaxBaseModelOutput:
        ...
    


class FlaxRobertaEncoder(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Unknown, ...] | FlaxBaseModelOutput:
        ...
    


class FlaxRobertaPooler(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states):
        ...
    


class FlaxRobertaLMHead(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    bias_init: Callable[..., np.ndarray] = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, shared_embedding=...):
        ...
    


class FlaxRobertaClassificationHead(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, deterministic=...):
        ...
    


class FlaxRobertaPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = RobertaConfig
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(self, config: RobertaConfig, input_shape: Tuple = ..., seed: int = ..., dtype: jnp.dtype = ..., **kwargs) -> None:
        ...
    
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        ...
    
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(self, input_ids, attention_mask=..., token_type_ids=..., position_ids=..., params: dict = ..., dropout_rng: PRNGKey = ..., train: bool = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...):
        ...
    


class FlaxRobertaModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    add_pooling_layer: bool = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown | Any, ...] | Any | FlaxBaseModelOutputWithPooling:
        ...
    


@add_start_docstrings("The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.", ROBERTA_START_DOCSTRING)
class FlaxRobertaModel(FlaxRobertaPreTrainedModel):
    module_class = ...


class FlaxRobertaForMaskedLMModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxMaskedLMOutput:
        ...
    


@add_start_docstrings("""RoBERTa Model with a `language modeling` head on top. """, ROBERTA_START_DOCSTRING)
class FlaxRobertaForMaskedLM(FlaxRobertaPreTrainedModel):
    module_class = ...


class FlaxRobertaForSequenceClassificationModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxSequenceClassifierOutput:
        ...
    


@add_start_docstrings("""
    Roberta Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """, ROBERTA_START_DOCSTRING)
class FlaxRobertaForSequenceClassification(FlaxRobertaPreTrainedModel):
    module_class = ...


class FlaxRobertaForMultipleChoiceModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxMultipleChoiceModelOutput:
        ...
    


@add_start_docstrings("""
    Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, ROBERTA_START_DOCSTRING)
class FlaxRobertaForMultipleChoice(FlaxRobertaPreTrainedModel):
    module_class = ...


class FlaxRobertaForTokenClassificationModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxTokenClassifierOutput:
        ...
    


@add_start_docstrings("""
    Roberta Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, ROBERTA_START_DOCSTRING)
class FlaxRobertaForTokenClassification(FlaxRobertaPreTrainedModel):
    module_class = ...


class FlaxRobertaForQuestionAnsweringModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxQuestionAnsweringModelOutput:
        ...
    


@add_start_docstrings("""
    Roberta Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, ROBERTA_START_DOCSTRING)
class FlaxRobertaForQuestionAnswering(FlaxRobertaPreTrainedModel):
    module_class = ...


