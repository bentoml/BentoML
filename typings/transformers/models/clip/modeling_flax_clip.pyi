

from typing import Any, Optional, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxlib.xla_extension as jax_xla
from flax.core.frozen_dict import FrozenDict

from ...file_utils import ModelOutput, add_start_docstrings
from ...modeling_flax_outputs import FlaxBaseModelOutputWithPooling
from ...modeling_flax_utils import FlaxPreTrainedModel
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig

logger = ...
CLIP_START_DOCSTRING = ...
CLIP_TEXT_INPUTS_DOCSTRING = ...
CLIP_VISION_INPUTS_DOCSTRING = ...
CLIP_INPUTS_DOCSTRING = ...
@flax.struct.dataclass
class FlaxCLIPOutput(ModelOutput):
    """
    Args:
        logits_per_image:(:obj:`jax_xla.DeviceArray` of shape :obj:`(image_batch_size, text_batch_size)`):
            The scaled dot product scores between :obj:`image_embeds` and :obj:`text_embeds`. This represents the
            image-text similarity scores.
        logits_per_text:(:obj:`jax_xla.DeviceArray` of shape :obj:`(text_batch_size, image_batch_size)`):
            The scaled dot product scores between :obj:`text_embeds` and :obj:`image_embeds`. This represents the
            text-image similarity scores.
        text_embeds(:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            :class:`~transformers.FlaxCLIPTextModel`.
        image_embeds(:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            :class:`~transformers.FlaxCLIPVisionModel`.
        text_model_output(:obj:`FlaxBaseModelOutputWithPooling`):
            The output of the :class:`~transformers.FlaxCLIPTextModel`.
        vision_model_output(:obj:`FlaxBaseModelOutputWithPooling`):
            The output of the :class:`~transformers.FlaxCLIPVisionModel`.
    """
    logits_per_image: jax_xla.DeviceArray = ...
    logits_per_text: jax_xla.DeviceArray = ...
    text_embeds: jax_xla.DeviceArray = ...
    image_embeds: jax_xla.DeviceArray = ...
    text_model_output: FlaxBaseModelOutputWithPooling = ...
    vision_model_output: FlaxBaseModelOutputWithPooling = ...
    def to_tuple(self) -> Tuple[Any]:
        ...
    


class FlaxCLIPVisionEmbeddings(nn.Module):
    config: CLIPVisionConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, pixel_values):
        ...
    


class FlaxCLIPTextEmbeddings(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, position_ids):
        ...
    


class FlaxCLIPAttention(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask=..., deterministic: bool = ..., output_attentions: bool = ...): # -> tuple[Unknown, Unknown] | tuple[Unknown]:
        ...
    


class FlaxCLIPMLP(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states):
        ...
    


class FlaxCLIPEncoderLayer(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic: bool = ..., output_attentions: bool = ...): # -> tuple[Unknown, Unknown] | tuple[Unknown]:
        ...
    


class FlaxCLIPLayerCollection(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask=..., deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Unknown, ...] | FlaxBaseModelOutput:
        ...
    


class FlaxCLIPEncoder(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, inputs_embeds, attention_mask=..., deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Unknown, ...] | FlaxBaseModelOutput:
        ...
    


class FlaxCLIPTextTransformer(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxBaseModelOutputWithPooling:
        ...
    


class FlaxCLIPVisionTransformer(nn.Module):
    config: CLIPVisionConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, pixel_values=..., deterministic: bool = ..., output_attentions=..., output_hidden_states=..., return_dict: bool = ...): # -> Tuple[Unknown | Any, ...] | Any | FlaxBaseModelOutputWithPooling:
        ...
    


class FlaxCLIPTextPreTrainedModel(FlaxPreTrainedModel):
    config_class = CLIPTextConfig
    module_class: nn.Module = ...
    def __init__(self, config: CLIPTextConfig, input_shape=..., seed: int = ..., dtype: jnp.dtype = ..., **kwargs) -> None:
        ...
    
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        ...
    
    def __call__(self, input_ids, attention_mask=..., position_ids=..., params: dict = ..., dropout_rng: jax.random.PRNGKey = ..., train: bool = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...):
        ...
    


class FlaxCLIPVisionPreTrainedModel(FlaxPreTrainedModel):
    config_class = CLIPVisionConfig
    module_class: nn.Module = ...
    def __init__(self, config: CLIPVisionConfig, input_shape: Optional[Tuple] = ..., seed: int = ..., dtype: jnp.dtype = ..., **kwargs) -> None:
        ...
    
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        ...
    
    def __call__(self, pixel_values, params: dict = ..., dropout_rng: jax.random.PRNGKey = ..., train: bool = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...):
        ...
    


class FlaxCLIPPreTrainedModel(FlaxPreTrainedModel):
    config_class = CLIPConfig
    module_class: nn.Module = ...
    def __init__(self, config: CLIPConfig, input_shape: Optional[Tuple] = ..., seed: int = ..., dtype: jnp.dtype = ..., **kwargs) -> None:
        ...
    
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        ...
    
    def __call__(self, input_ids, pixel_values, attention_mask=..., position_ids=..., params: dict = ..., dropout_rng: jax.random.PRNGKey = ..., train: bool = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...):
        ...
    
    def get_text_features(self, input_ids, attention_mask=..., position_ids=..., dropout_rng: jax.random.PRNGKey = ..., train=...):
        r"""
        Args:
            input_ids (:obj:`numpy.ndarray` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.CLIPTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__

        Returns:
            text_features (:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, output_dim`): The text embeddings
            obtained by applying the projection layer to the pooled output of :class:`~transformers.FlaxCLIPTextModel`.

        Examples::

            >>> from transformers import CLIPTokenizer, FlaxCLIPModel

            >>> model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

            >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"],  padding=True, return_tensors="np")
            >>> text_features = model.get_text_features(**inputs)
        """
        ...
    
    def get_image_features(self, pixel_values, dropout_rng: jax.random.PRNGKey = ..., train=...):
        r"""
        Args:
            pixel_values (:obj:`numpy.ndarray` of shape :obj:`(batch_size, num_channels, height, width)`):
                Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained
                using :class:`~transformers.CLIPFeatureExtractor`. See
                :meth:`transformers.CLIPFeatureExtractor.__call__` for details.

        Returns:
            image_features (:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, output_dim`): The image embeddings
            obtained by applying the projection layer to the pooled output of
            :class:`~transformers.FlaxCLIPVisionModel`

        Examples::

            >>> from PIL import Image
            >>> import requests
            >>> from transformers import CLIPProcessor, FlaxCLIPModel

            >>> model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> inputs = processor(images=image, return_tensors="np")

            >>> image_features = model.get_image_features(**inputs)
        """
        ...
    


class FlaxCLIPTextModule(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown, ...] | Any | FlaxBaseModelOutputWithPooling:
        ...
    


class FlaxCLIPTextModel(FlaxCLIPTextPreTrainedModel):
    module_class = ...


FLAX_CLIP_TEXT_MODEL_DOCSTRING = ...
class FlaxCLIPVisionModule(nn.Module):
    config: CLIPVisionConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, pixel_values, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> Tuple[Unknown | Any, ...] | Any | FlaxBaseModelOutputWithPooling:
        ...
    


class FlaxCLIPVisionModel(FlaxCLIPVisionPreTrainedModel):
    module_class = ...


FLAX_CLIP_VISION_MODEL_DOCSTRING = ...
class FlaxCLIPModule(nn.Module):
    config: CLIPConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids=..., pixel_values=..., attention_mask=..., position_ids=..., deterministic: bool = ..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> tuple[Unknown, Unknown, Unknown, Unknown, Tuple[Unknown, ...] | Unknown | Any | FlaxBaseModelOutputWithPooling, Tuple[Unknown | Any, ...] | Unknown | Any | FlaxBaseModelOutputWithPooling] | FlaxCLIPOutput:
        ...
    


@add_start_docstrings(CLIP_START_DOCSTRING)
class FlaxCLIPModel(FlaxCLIPPreTrainedModel):
    module_class = ...


FLAX_CLIP_MODEL_DOCSTRING = ...
