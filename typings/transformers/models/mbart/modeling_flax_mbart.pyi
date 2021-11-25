

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
from .configuration_mbart import MBartConfig

""" Flax MBart model. """
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
MBART_START_DOCSTRING = ...
MBART_INPUTS_DOCSTRING = ...
MBART_ENCODE_INPUTS_DOCSTRING = ...
MBART_DECODE_INPUTS_DOCSTRING = ...
def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int) -> jnp.ndarray:
    """
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    """
    ...

class FlaxMBartAttention(nn.Module):
    config: MBartConfig
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
    


class FlaxMBartEncoderLayer(nn.Module):
    config: MBartConfig
    dtype: jnp.dtype = ...
    def setup(self) -> None:
        ...
    
    def __call__(self, hidden_states: jnp.ndarray, attention_mask: jnp.ndarray, output_attentions: bool = ..., deterministic: bool = ...) -> Tuple[jnp.ndarray]:
        ...
    


class FlaxMBartEncoderLayerCollection(nn.Module):
    config: MBartConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Unknown | ndarray | tuple[()] | None, ...] | FlaxBaseModelOutput:
        ...
    


class FlaxMBartDecoderLayer(nn.Module):
    config: MBartConfig
    dtype: jnp.dtype = ...
    def setup(self) -> None:
        ...
    
    def __call__(self, hidden_states: jnp.ndarray, attention_mask: jnp.ndarray, encoder_hidden_states: Optional[jnp.ndarray] = ..., encoder_attention_mask: Optional[jnp.ndarray] = ..., init_cache: bool = ..., output_attentions: bool = ..., deterministic: bool = ...) -> Tuple[jnp.ndarray]:
        ...
    


class FlaxMBartDecoderLayerCollection(nn.Module):
    config: MBartConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, encoder_hidden_states: Optional[jnp.ndarray] = ..., encoder_attention_mask: Optional[jnp.ndarray] = ..., deterministic: bool = ..., init_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Unknown | ndarray | tuple[()] | None, ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...
    


class FlaxMBartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    config: MBartConfig
    inner_dim: int
    num_classes: int
    pooler_dropout: float
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states: jnp.ndarray, deterministic: bool): # -> ndarray:
        ...
    


class FlaxMBartEncoder(nn.Module):
    config: MBartConfig
    dtype: jnp.dtype = ...
    embed_tokens: Optional[nn.Embed] = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, position_ids, output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxBaseModelOutput:
        ...
    


class FlaxMBartDecoder(nn.Module):
    config: MBartConfig
    dtype: jnp.dtype = ...
    embed_tokens: Optional[nn.Embed] = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, position_ids, encoder_hidden_states: Optional[jnp.ndarray] = ..., encoder_attention_mask: Optional[jnp.ndarray] = ..., init_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...
    


class FlaxMBartModule(nn.Module):
    config: MBartConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, position_ids, decoder_position_ids, output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ...): # -> FlaxSeq2SeqModelOutput:
        ...
    


class FlaxMBartPreTrainedModel(FlaxPreTrainedModel):
    config_class = MBartConfig
    base_model_prefix: str = ...
    module_class: nn.Module = ...
    def __init__(self, config: MBartConfig, input_shape: Tuple[int] = ..., seed: int = ..., dtype: jnp.dtype = ..., **kwargs) -> None:
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
    
    @add_start_docstrings(MBART_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=MBartConfig)
    def encode(self, input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = ..., position_ids: Optional[jnp.ndarray] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...):
        r"""
        Returns:

        Example::

            >>> from transformers import MBartTokenizer, FlaxMBartForConditionalGeneration

            >>> model = FlaxMBartForConditionalGeneration.from_pretrained('facebook/mbart-large-cc25')
            >>> tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')

            >>> text = "My friends are cool but they eat too many carbs."
            >>> inputs = tokenizer(text, max_length=1024, return_tensors='jax')
            >>> encoder_outputs = model.encode(**inputs)
        """
        ...
    
    @add_start_docstrings(MBART_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=MBartConfig)
    def decode(self, decoder_input_ids, encoder_outputs, encoder_attention_mask: Optional[jnp.ndarray] = ..., decoder_attention_mask: Optional[jnp.ndarray] = ..., decoder_position_ids: Optional[jnp.ndarray] = ..., past_key_values: dict = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...):
        r"""
        Returns:

        Example::

            >>> from transformers import MBartTokenizer, FlaxMBartForConditionalGeneration

            >>> model = FlaxMBartForConditionalGeneration.from_pretrained('facebook/mbart-large-cc25')
            >>> tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')

            >>> text = "My friends are cool but they eat too many carbs."
            >>> inputs = tokenizer(text, max_length=1024, return_tensors='jax')
            >>> encoder_outputs = model.encode(**inputs)

            >>> decoder_start_token_id = model.config.decoder_start_token_id
            >>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

            >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
            >>> last_decoder_hidden_states = outputs.last_hidden_state
        """
        ...
    
    def __call__(self, input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = ..., decoder_input_ids: Optional[jnp.ndarray] = ..., decoder_attention_mask: Optional[jnp.ndarray] = ..., position_ids: Optional[jnp.ndarray] = ..., decoder_position_ids: Optional[jnp.ndarray] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...):
        ...
    


@add_start_docstrings("The bare MBart Model transformer outputting raw hidden-states without any specific head on top.", MBART_START_DOCSTRING)
class FlaxMBartModel(FlaxMBartPreTrainedModel):
    config: MBartConfig
    dtype: jnp.dtype = ...
    module_class = ...


class FlaxMBartForConditionalGenerationModule(nn.Module):
    config: MBartConfig
    dtype: jnp.dtype = ...
    bias_init: Callable[..., jnp.ndarray] = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, position_ids, decoder_position_ids, output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ...): # -> Any | FlaxSeq2SeqLMOutput:
        ...
    


@add_start_docstrings("The MMBart Model with a language modeling head. Can be used for summarization.", MBART_START_DOCSTRING)
class FlaxMBartForConditionalGeneration(FlaxMBartPreTrainedModel):
    module_class = ...
    dtype: jnp.dtype = ...
    @add_start_docstrings(MBART_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=MBartConfig)
    def decode(self, decoder_input_ids, encoder_outputs, encoder_attention_mask: Optional[jnp.ndarray] = ..., decoder_attention_mask: Optional[jnp.ndarray] = ..., decoder_position_ids: Optional[jnp.ndarray] = ..., past_key_values: dict = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., deterministic: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...): # -> FlaxCausalLMOutputWithCrossAttentions | Any:
        r"""
        Returns:

        Example::

            >>> from transformers import MBartTokenizer, FlaxMBartForConditionalGeneration

            >>> model = FlaxMBartForConditionalGeneration.from_pretrained('facebook/mbart-large-cc25')
            >>> tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')

            >>> text = "My friends are cool but they eat too many carbs."
            >>> inputs = tokenizer(text, max_length=1024, return_tensors='jax')
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
    


FLAX_MBART_CONDITIONAL_GENERATION_DOCSTRING = ...
class FlaxMBartForSequenceClassificationModule(nn.Module):
    config: MBartConfig
    dtype: jnp.dtype = ...
    num_labels: Optional[int] = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, position_ids, decoder_position_ids, output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ...): # -> Any | FlaxSeq2SeqSequenceClassifierOutput:
        ...
    


@add_start_docstrings("""
    MBart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """, MBART_START_DOCSTRING)
class FlaxMBartForSequenceClassification(FlaxMBartPreTrainedModel):
    module_class = ...
    dtype = ...


class FlaxMBartForQuestionAnsweringModule(nn.Module):
    config: MBartConfig
    dtype: jnp.dtype = ...
    num_labels = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, position_ids, decoder_position_ids, output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ...): # -> Any | FlaxSeq2SeqQuestionAnsweringModelOutput:
        ...
    


@add_start_docstrings("""
    MBart Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, MBART_START_DOCSTRING)
class FlaxMBartForQuestionAnswering(FlaxMBartPreTrainedModel):
    module_class = ...
    dtype = ...


