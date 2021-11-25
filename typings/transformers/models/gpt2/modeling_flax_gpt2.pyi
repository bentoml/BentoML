

from typing import Any, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_flax_utils import FlaxPreTrainedModel
from .configuration_gpt2 import GPT2Config

logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
GPT2_START_DOCSTRING = ...
GPT2_INPUTS_DOCSTRING = ...
class FlaxConv1D(nn.Module):
    features: int
    use_bias: bool = ...
    dtype: Any = ...
    precision: Any = ...
    @nn.compact
    def __call__(self, inputs): # -> Array:
        ...
    


class FlaxGPT2Attention(nn.Module):
    config: GPT2Config
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask=..., deterministic: bool = ..., init_cache: bool = ..., output_attentions: bool = ...): # -> tuple[Unknown, Unknown] | tuple[Unknown]:
        ...
    


class FlaxGPT2MLP(nn.Module):
    config: GPT2Config
    intermediate_size: int
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, deterministic: bool = ...):
        ...
    


class FlaxGPT2Block(nn.Module):
    config: GPT2Config
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask=..., deterministic: bool = ..., init_cache: bool = ..., output_attentions: bool = ...): # -> tuple[Unknown, Unknown] | tuple[Unknown]:
        ...
    


class FlaxGPT2PreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GPT2Config
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(self, config: GPT2Config, input_shape: Tuple = ..., seed: int = ..., dtype: jnp.dtype = ..., **kwargs) -> None:
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
    
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    def __call__(self, input_ids, attention_mask=..., position_ids=..., params: dict = ..., past_key_values: dict = ..., dropout_rng: jax.random.PRNGKey = ..., train: bool = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...):
        ...
    


class FlaxGPT2BlockCollection(nn.Module):
    config: GPT2Config
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask=..., deterministic: bool = ..., init_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Unknown, ...] | FlaxBaseModelOutputWithPast:
        ...
    


class FlaxGPT2Module(nn.Module):
    config: GPT2Config
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, position_ids, deterministic=..., init_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxBaseModelOutput:
        ...
    


@add_start_docstrings("The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.", GPT2_START_DOCSTRING)
class FlaxGPT2Model(FlaxGPT2PreTrainedModel):
    module_class = ...


class FlaxGPT2LMHeadModule(nn.Module):
    config: GPT2Config
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, position_ids, deterministic: bool = ..., init_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxCausalLMOutput:
        ...
    


@add_start_docstrings("""
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """, GPT2_START_DOCSTRING)
class FlaxGPT2LMHeadModel(FlaxGPT2PreTrainedModel):
    module_class = ...
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jnp.DeviceArray] = ...): # -> dict[str, Unknown | Array]:
        ...
    
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        ...
    


