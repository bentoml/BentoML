

from typing import Callable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxlib.xla_extension as jax_xla
import numpy as np
from flax.core.frozen_dict import FrozenDict
from jax.random import PRNGKey

from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_flax_utils import FlaxPreTrainedModel
from .configuration_electra import ElectraConfig

logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
@flax.struct.dataclass
class FlaxElectraForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.ElectraForPreTraining`.

    Args:
        logits (:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
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
    logits: jax_xla.DeviceArray = ...
    hidden_states: Optional[Tuple[jax_xla.DeviceArray]] = ...
    attentions: Optional[Tuple[jax_xla.DeviceArray]] = ...


ELECTRA_START_DOCSTRING = ...
ELECTRA_INPUTS_DOCSTRING = ...
class FlaxElectraEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    config: ElectraConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = ...):
        ...
    


class FlaxElectraSelfAttention(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic=..., output_attentions: bool = ...): # -> tuple[Unknown, Unknown] | tuple[Unknown]:
        ...
    


class FlaxElectraSelfOutput(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, input_tensor, deterministic: bool = ...):
        ...
    


class FlaxElectraAttention(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic=..., output_attentions: bool = ...): # -> tuple[Unknown, Unknown] | tuple[Unknown]:
        ...
    


class FlaxElectraIntermediate(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states):
        ...
    


class FlaxElectraOutput(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_output, deterministic: bool = ...):
        ...
    


class FlaxElectraLayer(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic: bool = ..., output_attentions: bool = ...): # -> tuple[Unknown, Unknown] | tuple[Unknown]:
        ...
    


class FlaxElectraLayerCollection(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Unknown, ...] | FlaxBaseModelOutput:
        ...
    


class FlaxElectraEncoder(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Unknown, ...] | FlaxBaseModelOutput:
        ...
    


class FlaxElectraGeneratorPredictions(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states):
        ...
    


class FlaxElectraDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""
    config: ElectraConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states):
        ...
    


class FlaxElectraPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ElectraConfig
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(self, config: ElectraConfig, input_shape: Tuple = ..., seed: int = ..., dtype: jnp.dtype = ..., **kwargs) -> None:
        ...
    
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        ...
    
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(self, input_ids, attention_mask=..., token_type_ids=..., position_ids=..., params: dict = ..., dropout_rng: PRNGKey = ..., train: bool = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...):
        ...
    


class FlaxElectraModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Unknown, ...] | FlaxBaseModelOutput:
        ...
    


@add_start_docstrings("The bare Electra Model transformer outputting raw hidden-states without any specific head on top.", ELECTRA_START_DOCSTRING)
class FlaxElectraModel(FlaxElectraPreTrainedModel):
    module_class = ...


class FlaxElectraTiedDense(nn.Module):
    embedding_size: int
    dtype: jnp.dtype = ...
    precision = ...
    bias_init: Callable[..., np.ndarray] = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, x, kernel):
        ...
    


class FlaxElectraForMaskedLMModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask=..., token_type_ids=..., position_ids=..., deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxMaskedLMOutput:
        ...
    


@add_start_docstrings("""Electra Model with a `language modeling` head on top. """, ELECTRA_START_DOCSTRING)
class FlaxElectraForMaskedLM(FlaxElectraPreTrainedModel):
    module_class = ...


class FlaxElectraForPreTrainingModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask=..., token_type_ids=..., position_ids=..., deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxElectraForPreTrainingOutput:
        ...
    


@add_start_docstrings("""
    Electra model with a binary classification head on top as used during pretraining for identifying generated tokens.

    It is recommended to load the discriminator checkpoint into that model.
    """, ELECTRA_START_DOCSTRING)
class FlaxElectraForPreTraining(FlaxElectraPreTrainedModel):
    module_class = ...


FLAX_ELECTRA_FOR_PRETRAINING_DOCSTRING = ...
class FlaxElectraForTokenClassificationModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask=..., token_type_ids=..., position_ids=..., deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxTokenClassifierOutput:
        ...
    


@add_start_docstrings("""
    Electra model with a token classification head on top.

    Both the discriminator and generator may be loaded into this model.
    """, ELECTRA_START_DOCSTRING)
class FlaxElectraForTokenClassification(FlaxElectraPreTrainedModel):
    module_class = ...


def identity(x, **kwargs):
    ...

class FlaxElectraSequenceSummary(nn.Module):
    r"""
    Compute a single vector summary of a sequence hidden states.

    Args:
        config (:class:`~transformers.PretrainedConfig`):
            The config used by the model. Relevant arguments in the config class of the model are (refer to the actual
            config class of your model for the default values it uses):

            - **summary_use_proj** (:obj:`bool`) -- Add a projection after the vector extraction.
            - **summary_proj_to_labels** (:obj:`bool`) -- If :obj:`True`, the projection outputs to
              :obj:`config.num_labels` classes (otherwise to :obj:`config.hidden_size`).
            - **summary_activation** (:obj:`Optional[str]`) -- Set to :obj:`"tanh"` to add a tanh activation to the
              output, another string or :obj:`None` will add no activation.
            - **summary_first_dropout** (:obj:`float`) -- Optional dropout probability before the projection and
              activation.
            - **summary_last_dropout** (:obj:`float`)-- Optional dropout probability after the projection and
              activation.
    """
    config: ElectraConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, cls_index=..., deterministic: bool = ...):
        """
        Compute a single vector summary of a sequence hidden states.

        Args:
            hidden_states (:obj:`jnp.array` of shape :obj:`[batch_size, seq_len, hidden_size]`):
                The hidden states of the last layer.
            cls_index (:obj:`jnp.array` of shape :obj:`[batch_size]` or :obj:`[batch_size, ...]` where ... are optional leading dimensions of :obj:`hidden_states`, `optional`):
                Used if :obj:`summary_type == "cls_index"` and takes the last token of the sequence as classification
                token.

        Returns:
            :obj:`jnp.array`: The summary of the sequence hidden states.
        """
        ...
    


class FlaxElectraForMultipleChoiceModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask=..., token_type_ids=..., position_ids=..., deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxMultipleChoiceModelOutput:
        ...
    


@add_start_docstrings("""
    ELECTRA Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, ELECTRA_START_DOCSTRING)
class FlaxElectraForMultipleChoice(FlaxElectraPreTrainedModel):
    module_class = ...


class FlaxElectraForQuestionAnsweringModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask=..., token_type_ids=..., position_ids=..., deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxQuestionAnsweringModelOutput:
        ...
    


@add_start_docstrings("""
    ELECTRA Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, ELECTRA_START_DOCSTRING)
class FlaxElectraForQuestionAnswering(FlaxElectraPreTrainedModel):
    module_class = ...


class FlaxElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    config: ElectraConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, deterministic: bool = ...):
        ...
    


class FlaxElectraForSequenceClassificationModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask=..., token_type_ids=..., position_ids=..., deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxSequenceClassifierOutput:
        ...
    


@add_start_docstrings("""
    Electra Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """, ELECTRA_START_DOCSTRING)
class FlaxElectraForSequenceClassification(FlaxElectraPreTrainedModel):
    module_class = ...


