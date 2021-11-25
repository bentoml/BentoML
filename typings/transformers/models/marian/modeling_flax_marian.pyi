

import random
from typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax.random import PRNGKey

from ...file_utils import add_start_docstrings, replace_return_docstrings
from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
)
from ...modeling_flax_utils import FlaxPreTrainedModel
from .configuration_marian import MarianConfig

""" Flax Marian model. """
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
MARIAN_START_DOCSTRING = ...
MARIAN_INPUTS_DOCSTRING = ...
MARIAN_ENCODE_INPUTS_DOCSTRING = ...
MARIAN_DECODE_INPUTS_DOCSTRING = ...
def create_sinusoidal_positions(n_pos, dim, dtype):
    ...

def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    Shift input ids one token to the right.
    """
    ...

class FlaxMarianAttention(nn.Module):
    config: MarianConfig
    embed_dim: int
    num_heads: int
    dropout: float = ...
    causal: bool = ...
    bias: bool = ...
    dtype: jnp.dtype = ...
    def setup(self) -> None:
        ...
    
    def __call__(self, hidden_states: jnp.ndarray, key_value_states: Optional[jnp.ndarray] = ..., attention_mask: Optional[jnp.ndarray] = ..., init_cache: bool = ..., deterministic: bool = ...) -> Tuple[jnp.ndarray]:
        """Input shape: Batch x Time x Channel"""
        ...
    


class FlaxMarianEncoderLayer(nn.Module):
    config: MarianConfig
    dtype: jnp.dtype = ...
    def setup(self) -> None:
        ...
    
    def __call__(self, hidden_states: jnp.ndarray, attention_mask: jnp.ndarray, output_attentions: bool = ..., deterministic: bool = ...) -> Tuple[jnp.ndarray]:
        ...
    


class FlaxMarianEncoderLayerCollection(nn.Module):
    config: MarianConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Unknown | ndarray | tuple[()] | None, ...] | FlaxBaseModelOutput:
        ...
    


class FlaxMarianDecoderLayer(nn.Module):
    config: MarianConfig
    dtype: jnp.dtype = ...
    def setup(self) -> None:
        ...
    
    def __call__(self, hidden_states: jnp.ndarray, attention_mask: jnp.ndarray, encoder_hidden_states: Optional[jnp.ndarray] = ..., encoder_attention_mask: Optional[jnp.ndarray] = ..., init_cache: bool = ..., output_attentions: bool = ..., deterministic: bool = ...) -> Tuple[jnp.ndarray]:
        ...
    


class FlaxMarianDecoderLayerCollection(nn.Module):
    config: MarianConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, encoder_hidden_states: Optional[jnp.ndarray] = ..., encoder_attention_mask: Optional[jnp.ndarray] = ..., deterministic: bool = ..., init_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Unknown | ndarray | tuple[()] | None, ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...
    


class FlaxMarianEncoder(nn.Module):
    config: MarianConfig
    dtype: jnp.dtype = ...
    embed_tokens: Optional[nn.Embed] = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, position_ids, output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ...): # -> tuple[Unknown | ndarray | tuple[()] | None, ...] | FlaxBaseModelOutput:
        ...
    


class FlaxMarianDecoder(nn.Module):
    config: MarianConfig
    dtype: jnp.dtype = ...
    embed_tokens: Optional[nn.Embed] = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, position_ids, encoder_hidden_states: Optional[jnp.ndarray] = ..., encoder_attention_mask: Optional[jnp.ndarray] = ..., init_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ...): # -> tuple[Unknown | ndarray | tuple[()] | None, ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...
    


class FlaxMarianModule(nn.Module):
    config: MarianConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, position_ids, decoder_position_ids, output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ...): # -> FlaxSeq2SeqModelOutput:
        ...
    


class FlaxMarianPreTrainedModel(FlaxPreTrainedModel):
    config_class = MarianConfig
    base_model_prefix: str = ...
    module_class: nn.Module = ...
    def __init__(self, config: MarianConfig, input_shape: Tuple[int] = ..., seed: int = ..., dtype: jnp.dtype = ..., **kwargs) -> None:
        ...
    
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        ...
    
    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (:obj:`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (:obj:`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
            encoder_outputs (:obj:`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                ``encoder_outputs`` consists of (:obj:`last_hidden_state`, `optional`: :obj:`hidden_states`,
                `optional`: :obj:`attentions`). :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length,
                hidden_size)`, `optional`) is a sequence of hidden-states at the output of the last layer of the
                encoder. Used in the cross-attention of the decoder.
        """
        ...
    
    @add_start_docstrings(MARIAN_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=MarianConfig)
    def encode(self, input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = ..., position_ids: Optional[jnp.ndarray] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...):
        r"""
        Returns:

        Example::

            >>> from transformers import MarianTokenizer, FlaxMarianMTModel

            >>> tokenizer = MarianTokenizer.from_pretrained('facebook/marian-large-cnn')
            >>> model = FlaxMarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')

            >>> text = "My friends are cool but they eat too many carbs."
            >>> inputs = tokenizer(text, max_length=64, return_tensors='jax')
            >>> encoder_outputs = model.encode(**inputs)
        """
        ...
    
    @add_start_docstrings(MARIAN_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=MarianConfig)
    def decode(self, decoder_input_ids, encoder_outputs, encoder_attention_mask: Optional[jnp.ndarray] = ..., decoder_attention_mask: Optional[jnp.ndarray] = ..., decoder_position_ids: Optional[jnp.ndarray] = ..., past_key_values: dict = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...):
        r"""
        Returns:

        Example::

            >>> from transformers import MarianTokenizer, FlaxMarianMTModel

            >>> tokenizer = MarianTokenizer.from_pretrained('facebook/marian-large-cnn')
            >>> model = FlaxMarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')

            >>> text = "My friends are cool but they eat too many carbs."
            >>> inputs = tokenizer(text, max_length=64, return_tensors='jax')
            >>> encoder_outputs = model.encode(**inputs)

            >>> decoder_start_token_id = model.config.decoder_start_token_id
            >>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

            >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
            >>> last_decoder_hidden_states = outputs.last_hidden_state
        """
        ...
    
    def __call__(self, input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = ..., decoder_input_ids: Optional[jnp.ndarray] = ..., decoder_attention_mask: Optional[jnp.ndarray] = ..., position_ids: Optional[jnp.ndarray] = ..., decoder_position_ids: Optional[jnp.ndarray] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...):
        ...
    


@add_start_docstrings("The bare Marian Model transformer outputting raw hidden-states without any specific head on top.", MARIAN_START_DOCSTRING)
class FlaxMarianModel(FlaxMarianPreTrainedModel):
    config: MarianConfig
    dtype: jnp.dtype = ...
    module_class = ...


class FlaxMarianMTModule(nn.Module):
    config: MarianConfig
    dtype: jnp.dtype = ...
    bias_init: Callable[..., jnp.ndarray] = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, position_ids, decoder_position_ids, output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ...): # -> Any | FlaxSeq2SeqLMOutput:
        ...
    


@add_start_docstrings("The MARIAN Model with a language modeling head. Can be used for translation.", MARIAN_START_DOCSTRING)
class FlaxMarianMTModel(FlaxMarianPreTrainedModel):
    module_class = ...
    dtype: jnp.dtype = ...
    @add_start_docstrings(MARIAN_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=MarianConfig)
    def decode(self, decoder_input_ids, encoder_outputs, encoder_attention_mask: Optional[jnp.ndarray] = ..., decoder_attention_mask: Optional[jnp.ndarray] = ..., decoder_position_ids: Optional[jnp.ndarray] = ..., past_key_values: dict = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., deterministic: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...): # -> FlaxCausalLMOutputWithCrossAttentions | Any:
        r"""
        Returns:

        Example::

            >>> from transformers import MarianTokenizer, FlaxMarianMTModel

            >>> model = FlaxMarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')
            >>> tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')

            >>> text = "My friends are cool but they eat too many carbs."
            >>> inputs = tokenizer(text, max_length=64, return_tensors='jax')
            >>> encoder_outputs = model.encode(**inputs)

            >>> decoder_start_token_id = model.config.decoder_start_token_id
            >>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

            >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
            >>> logits = outputs.logits
        """
        ...
    
    def prepare_inputs_for_generation(self, decoder_input_ids, max_length, attention_mask: Optional[jnp.DeviceArray] = ..., decoder_attention_mask: Optional[jnp.DeviceArray] = ..., encoder_outputs=..., **kwargs): # -> dict[str, Unknown | DeviceArray | Array | None]:
        ...
    
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        ...
    


FLAX_MARIAN_MT_DOCSTRING = ...
