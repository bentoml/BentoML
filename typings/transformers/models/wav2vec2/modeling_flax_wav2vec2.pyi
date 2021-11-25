

from typing import Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_flax_utils import FlaxPreTrainedModel
from .configuration_wav2vec2 import Wav2Vec2Config

""" Flax Wav2Vec2 model. """
logger = ...
@flax.struct.dataclass
class FlaxWav2Vec2BaseModelOutput(ModelOutput):
    """
    Output type of :class:`~transformers.FlaxWav2Vec2BaseModelOutput`, with potential hidden states and attentions.

    Args:
        last_hidden_state (:obj:`jnp.ndarray` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        extract_features (:obj:`jnp.ndarray` of shape :obj:`(batch_size, sequence_length, last_conv_dim)`):
            Sequence of extracted feature vectors of the last convolutional layer of the model with ``last_conv_dim``
            being the dimension of the last convolutional layer.
        hidden_states (:obj:`tuple(jnp.ndarray)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(jnp.ndarray)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`jnp.ndarray` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    last_hidden_state: jnp.ndarray = ...
    extract_features: jnp.ndarray = ...
    hidden_states: Optional[Tuple[jnp.ndarray]] = ...
    attentions: Optional[Tuple[jnp.ndarray]] = ...


@flax.struct.dataclass
class FlaxWav2Vec2ForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.FlaxWav2Vec2ForPreTrainingOutput`, with potential hidden states and
    attentions.

    Args:
        loss (`optional`, returned when model is in train mode, ``jnp.ndarray`` of shape :obj:`(1,)`):
            Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the `official
            paper <https://arxiv.org/pdf/2006.11477.pdf>`__ . (classification) loss.
        projected_states (:obj:`jnp.ndarray` of shape :obj:`(batch_size, sequence_length, config.proj_codevector_dim)`):
            Hidden-states of the model projected to `config.proj_codevector_dim` that can be used to predict the masked
            projected quantized states.
        projected_quantized_states (:obj:`jnp.ndarray` of shape :obj:`(batch_size, sequence_length, config.proj_codevector_dim)`):
            Quantized extracted feature vectors projected to `config.proj_codevector_dim` representing the positive
            target vectors for contrastive loss.
        hidden_states (:obj:`tuple(jnp.ndarray)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(jnp.ndarray)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`jnp.ndarray` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    projected_states: jnp.ndarray = ...
    projected_quantized_states: jnp.ndarray = ...
    codevector_perplexity: jnp.ndarray = ...
    hidden_states: Optional[Tuple[jnp.ndarray]] = ...
    attentions: Optional[Tuple[jnp.ndarray]] = ...


WAV_2_VEC_2_START_DOCSTRING = ...
WAV_2_VEC_2_INPUTS_DOCSTRING = ...
class FlaxWav2Vec2LayerNormConvLayer(nn.Module):
    config: Wav2Vec2Config
    layer_id: int = ...
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states):
        ...
    


class FlaxConvWithWeightNorm(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states):
        ...
    


class FlaxWav2Vec2PositionalConvEmbedding(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states):
        ...
    


class FlaxConvLayersCollection(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states):
        ...
    


class FlaxWav2Vec2FeatureExtractor(nn.Module):
    """Construct the featurs from raw audio waveform"""
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_values):
        ...
    


class FlaxWav2Vec2FeatureProjection(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, deterministic=...): # -> tuple[Unknown, Unknown]:
        ...
    


class FlaxWav2Vec2Attention(nn.Module):
    config: Wav2Vec2Config
    embed_dim: int
    num_heads: int
    dropout: float = ...
    bias: bool = ...
    dtype: jnp.dtype = ...
    def setup(self) -> None:
        ...
    
    def __call__(self, hidden_states: jnp.ndarray, key_value_states: Optional[jnp.ndarray] = ..., attention_mask: Optional[jnp.ndarray] = ..., deterministic: bool = ...) -> Tuple[jnp.ndarray]:
        """Input shape: Batch x Time x Channel"""
        ...
    


class FlaxWav2Vec2FeedForward(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, deterministic=...):
        ...
    


class FlaxWav2Vec2EncoderLayerStableLayerNorm(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask=..., deterministic=..., output_attentions=...): # -> tuple[Unknown, Unknown] | tuple[Unknown]:
        ...
    


class FlaxWav2Vec2EncoderLayerStableLayerNormCollection(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask=..., deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Unknown, ...] | FlaxBaseModelOutput:
        ...
    


class FlaxWav2Vec2StableLayerNormEncoder(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask=..., deterministic=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Tuple[Unknown, ...] | Any | FlaxBaseModelOutput:
        ...
    


class FlaxWav2Vec2GumbelVectorQuantizer(nn.Module):
    """
    Vector quantization using gumbel softmax. See `CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX
    <https://arxiv.org/pdf/1611.01144.pdf>`__ for more information.
    """
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, mask_time_indices=..., deterministic=..., temperature=...): # -> tuple[Unknown, Unknown]:
        ...
    


class FlaxWav2Vec2PreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Wav2Vec2Config
    base_model_prefix: str = ...
    module_class: nn.Module = ...
    def __init__(self, config: Wav2Vec2Config, input_shape: Tuple = ..., seed: int = ..., dtype: jnp.dtype = ..., **kwargs) -> None:
        ...
    
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        ...
    
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    def __call__(self, input_values, attention_mask=..., mask_time_indices=..., params: dict = ..., dropout_rng: jax.random.PRNGKey = ..., train: bool = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...):
        ...
    


class FlaxWav2Vec2Module(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_values, attention_mask=..., mask_time_indices=..., deterministic=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Tuple[Unknown | Any, ...] | Any | FlaxWav2Vec2BaseModelOutput:
        """

        Returns:

        Example::

            >>> from transformers import Wav2Vec2Processor, FlaxWav2Vec2Model
            >>> from datasets import load_dataset
            >>> import soundfile as sf

            >>> processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            >>> model = FlaxWav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

            >>> def map_to_array(batch):
            >>>     speech, _ = sf.read(batch["file"])
            >>>     batch["speech"] = speech
            >>>     return batch

            >>> ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
            >>> ds = ds.map(map_to_array)

            >>> input_values = processor(ds["speech"][0], return_tensors="np").input_values  # Batch size 1
            >>> hidden_states = model(input_values).last_hidden_state

        """
        ...
    


@add_start_docstrings("The bare Wav2Vec2 Model transformer outputting raw hidden-states without any specific head on top.", WAV_2_VEC_2_START_DOCSTRING)
class FlaxWav2Vec2Model(FlaxWav2Vec2PreTrainedModel):
    module_class = ...


class FlaxWav2Vec2ForCTCModule(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_values, attention_mask=..., mask_time_indices=..., deterministic=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Tuple[Unknown, ...] | Any | FlaxCausalLMOutput:
        r"""
        Returns:

        Example::

            >>> import jax.numpy as jnp
            >>> from transformers import Wav2Vec2Processor, FlaxWav2Vec2ForCTC
            >>> from datasets import load_dataset
            >>> import soundfile as sf

            >>> processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            >>> model = FlaxWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

            >>> def map_to_array(batch):
            >>>     speech, _ = sf.read(batch["file"])
            >>>     batch["speech"] = speech
            >>>     return batch

            >>> ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
            >>> ds = ds.map(map_to_array)

            >>> input_values = processor(ds["speech"][0], return_tensors="np").input_values  # Batch size 1
            >>> logits = model(input_values).logits
            >>> predicted_ids = jnp.argmax(logits, axis=-1)

            >>> transcription = processor.decode(predicted_ids[0])
            >>> # should give:  "A MAN SAID TO THE UNIVERSE SIR I EXIST"

        """
        ...
    


@add_start_docstrings("Wav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).", WAV_2_VEC_2_START_DOCSTRING)
class FlaxWav2Vec2ForCTC(FlaxWav2Vec2PreTrainedModel):
    module_class = ...


class FlaxWav2Vec2ForPreTrainingModule(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_values, attention_mask=..., mask_time_indices=..., gumbel_temperature: int = ..., deterministic: bool = ..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Tuple[Unknown, ...] | Any | FlaxWav2Vec2ForPreTrainingOutput:
        r"""
        Returns:

        Example::

            >>> import optax
            >>> import numpy as np
            >>> import jax.numpy as jnp
            >>> from transformers import Wav2Vec2FeatureExtractor, FlaxWav2Vec2ForPreTraining
            >>> from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
            >>> from datasets import load_dataset
            >>> import soundfile as sf

            >>> feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("patrickvonplaten/wav2vec2-base")
            >>> model = FlaxWav2Vec2ForPreTraining.from_pretrained("patrickvonplaten/wav2vec2-base")


            >>> def map_to_array(batch):
            ...     speech, _ = sf.read(batch["file"])
            ...     batch["speech"] = speech
            ...     return batch


            >>> ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
            >>> ds = ds.map(map_to_array)

            >>> input_values = feature_extractor(ds["speech"][0], return_tensors="np").input_values  # Batch size 1

            >>> # compute masked indices
            >>> batch_size, raw_sequence_length = input_values.shape
            >>> sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)
            >>> mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.2, mask_length=2)

            >>> outputs = model(input_values, mask_time_indices=mask_time_indices)

            >>> # compute cosine similarity between predicted (=projected_states) and target (=projected_quantized_states)
            >>> cosine_sim = optax.cosine_similarity(
            ...     outputs.projected_states, outputs.projected_quantized_states, axis=-1
            ... )

            >>> # show that cosine similarity is much higher than random
            >>> assert np.asarray(cosine_sim)[mask_time_indices].mean() > 0.5

        """
        ...
    


@add_start_docstrings("""Wav2Vec2 Model with a quantizer and `VQ` head on top. """, WAV_2_VEC_2_START_DOCSTRING)
class FlaxWav2Vec2ForPreTraining(FlaxWav2Vec2PreTrainedModel):
    module_class = ...
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    def __call__(self, input_values, attention_mask=..., mask_time_indices=..., gumbel_temperature: int = ..., params: dict = ..., dropout_rng: jax.random.PRNGKey = ..., gumbel_rng: jax.random.PRNGKey = ..., train: bool = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...):
        ...
    


