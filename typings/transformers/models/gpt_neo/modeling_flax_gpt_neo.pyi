

from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_flax_utils import FlaxPreTrainedModel
from .configuration_gpt_neo import GPTNeoConfig

logger = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
GPT_NEO_START_DOCSTRING = ...
GPT_NEO_INPUTS_DOCSTRING = ...
class FlaxGPTNeoSelfAttention(nn.Module):
    config: GPTNeoConfig
    attention_type: str
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask=..., deterministic: bool = ..., init_cache: bool = ..., output_attentions: bool = ...): # -> tuple[Unknown, Unknown] | tuple[Unknown]:
        ...
    


class FlaxGPTNeoAttention(nn.Module):
    config: GPTNeoConfig
    layer_id: int = ...
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask=..., deterministic: bool = ..., init_cache: bool = ..., output_attentions: bool = ...): # -> tuple[Unknown, Unknown] | tuple[Unknown]:
        ...
    


class FlaxGPTNeoMLP(nn.Module):
    config: GPTNeoConfig
    intermediate_size: int
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, deterministic: bool = ...):
        ...
    


class FlaxGPTNeoBlock(nn.Module):
    config: GPTNeoConfig
    layer_id: int = ...
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask=..., deterministic: bool = ..., init_cache: bool = ..., output_attentions: bool = ...): # -> tuple[Unknown, Unknown] | tuple[Unknown]:
        ...
    


class FlaxGPTNeoPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GPTNeoConfig
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(self, config: GPTNeoConfig, input_shape: Tuple = ..., seed: int = ..., dtype: jnp.dtype = ..., **kwargs) -> None:
        ...
    
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        ...
    
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (:obj:`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (:obj:`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        ...
    
    @add_start_docstrings_to_model_forward(GPT_NEO_INPUTS_DOCSTRING)
    def __call__(self, input_ids, attention_mask=..., position_ids=..., params: dict = ..., past_key_values: dict = ..., dropout_rng: jax.random.PRNGKey = ..., train: bool = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...):
        ...
    


class FlaxGPTNeoBlockCollection(nn.Module):
    config: GPTNeoConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask=..., deterministic: bool = ..., init_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Unknown, ...] | FlaxBaseModelOutputWithPast:
        ...
    


class FlaxGPTNeoModule(nn.Module):
    config: GPTNeoConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, position_ids, deterministic=..., init_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxBaseModelOutput:
        ...
    


@add_start_docstrings("The bare GPTNeo Model transformer outputting raw hidden-states without any specific head on top.", GPT_NEO_START_DOCSTRING)
class FlaxGPTNeoModel(FlaxGPTNeoPreTrainedModel):
    module_class = ...


class FlaxGPTNeoForCausalLMModule(nn.Module):
    config: GPTNeoConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, position_ids, deterministic: bool = ..., init_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxCausalLMOutput:
        ...
    


@add_start_docstrings("""
    The GPTNeo Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """, GPT_NEO_START_DOCSTRING)
class FlaxGPTNeoForCausalLM(FlaxGPTNeoPreTrainedModel):
    module_class = ...
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jnp.DeviceArray] = ...): # -> dict[str, Unknown | Array]:
        ...
    
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        ...
    


