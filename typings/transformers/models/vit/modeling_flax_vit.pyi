

from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_flax_utils import FlaxPreTrainedModel
from .configuration_vit import ViTConfig

VIT_START_DOCSTRING = ...
VIT_INPUTS_DOCSTRING = ...
class FlaxPatchEmbeddings(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, pixel_values):
        ...
    


class FlaxViTEmbeddings(nn.Module):
    """Construct the CLS token, position and patch embeddings."""
    config: ViTConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, pixel_values, deterministic=...):
        ...
    


class FlaxViTSelfAttention(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, deterministic: bool = ..., output_attentions: bool = ...): # -> tuple[Unknown, Unknown] | tuple[Unknown]:
        ...
    


class FlaxViTSelfOutput(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, input_tensor, deterministic: bool = ...):
        ...
    


class FlaxViTAttention(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, deterministic=..., output_attentions: bool = ...): # -> tuple[Unknown, Unknown] | tuple[Unknown]:
        ...
    


class FlaxViTIntermediate(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states):
        ...
    


class FlaxViTOutput(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_output, deterministic: bool = ...):
        ...
    


class FlaxViTLayer(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, deterministic: bool = ..., output_attentions: bool = ...): # -> tuple[Unknown, Unknown] | tuple[Unknown]:
        ...
    


class FlaxViTLayerCollection(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Unknown, ...] | FlaxBaseModelOutput:
        ...
    


class FlaxViTEncoder(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Unknown, ...] | FlaxBaseModelOutput:
        ...
    


class FlaxViTPooler(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states):
        ...
    


class FlaxViTPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ViTConfig
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(self, config: ViTConfig, input_shape=..., seed: int = ..., dtype: jnp.dtype = ..., **kwargs) -> None:
        ...
    
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        ...
    
    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(self, pixel_values, params: dict = ..., dropout_rng: jax.random.PRNGKey = ..., train: bool = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...):
        ...
    


class FlaxViTModule(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = ...
    add_pooling_layer: bool = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, pixel_values, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxBaseModelOutputWithPooling:
        ...
    


@add_start_docstrings("The bare ViT Model transformer outputting raw hidden-states without any specific head on top.", VIT_START_DOCSTRING)
class FlaxViTModel(FlaxViTPreTrainedModel):
    module_class = ...


FLAX_VISION_MODEL_DOCSTRING = ...
class FlaxViTForImageClassificationModule(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, pixel_values=..., deterministic: bool = ..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Tuple[Unknown, ...] | Any | FlaxSequenceClassifierOutput:
        ...
    


@add_start_docstrings("""
    ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """, VIT_START_DOCSTRING)
class FlaxViTForImageClassification(FlaxViTPreTrainedModel):
    module_class = ...


FLAX_VISION_CLASSIF_DOCSTRING = ...
