

from typing import Any, Optional, Tuple

import torch
from torch import nn

from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig

""" PyTorch CLIP model. """
logger = ...
_CHECKPOINT_FOR_DOC = ...
CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = ...
def contrastive_loss(logits: torch.Tensor, dim: int) -> torch.Tensor:
    ...

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    ...

class CLIPOutput(ModelOutput):
    """
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`return_loss` is :obj:`True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(:obj:`torch.FloatTensor` of shape :obj:`(image_batch_size, text_batch_size)`):
            The scaled dot product scores between :obj:`image_embeds` and :obj:`text_embeds`. This represents the
            image-text similarity scores.
        logits_per_text:(:obj:`torch.FloatTensor` of shape :obj:`(text_batch_size, image_batch_size)`):
            The scaled dot product scores between :obj:`text_embeds` and :obj:`image_embeds`. This represents the
            text-image similarity scores.
        text_embeds(:obj:`torch.FloatTensor` of shape :obj:`(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            :class:`~transformers.CLIPTextModel`.
        image_embeds(:obj:`torch.FloatTensor` of shape :obj:`(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            :class:`~transformers.CLIPVisionModel`.
        text_model_output(:obj:`BaseModelOutputWithPooling`):
            The output of the :class:`~transformers.CLIPTextModel`.
        vision_model_output(:obj:`BaseModelOutputWithPooling`):
            The output of the :class:`~transformers.CLIPVisionModel`.
    """
    loss: Optional[torch.FloatTensor] = ...
    logits_per_image: torch.FloatTensor = ...
    logits_per_text: torch.FloatTensor = ...
    text_embeds: torch.FloatTensor = ...
    image_embeds: torch.FloatTensor = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> Tuple[Any]:
        ...
    


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig) -> None:
        ...
    
    def forward(self, pixel_values): # -> Any:
        ...
    


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig) -> None:
        ...
    
    def forward(self, input_ids=..., position_ids=..., inputs_embeds=...): # -> Any:
        ...
    


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., causal_attention_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        ...
    


class CLIPMLP(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, causal_attention_mask: torch.Tensor, output_attentions: bool = ...): # -> tuple[Tensor, Any] | tuple[Tensor]:
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape :obj:`(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                :obj:`(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                :obj:`(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        ...
    


class CLIPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = CLIPConfig
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...


CLIP_START_DOCSTRING = ...
CLIP_TEXT_INPUTS_DOCSTRING = ...
CLIP_VISION_INPUTS_DOCSTRING = ...
CLIP_INPUTS_DOCSTRING = ...
class CLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of :obj:`config.num_hidden_layers` self attention layers. Each layer is a
    :class:`~transformers.CLIPEncoderLayer`.

    Args:
        config: CLIPConfig
        embed_tokens (nn.Embedding): output embedding
    """
    def __init__(self, config: CLIPConfig) -> None:
        ...
    
    def forward(self, inputs_embeds, attention_mask=..., causal_attention_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        Args:
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            causal_attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Causal mask for the text model. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        ...
    


class CLIPTextTransformer(nn.Module):
    def __init__(self, config: CLIPTextConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
    def forward(self, input_ids=..., attention_mask=..., position_ids=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | BaseModelOutputWithPooling:
        r"""
        Returns:

        """
        ...
    


class CLIPTextModel(CLIPPreTrainedModel):
    config_class = CLIPTextConfig
    def __init__(self, config: CLIPTextConfig) -> None:
        ...
    
    def get_input_embeddings(self) -> nn.Module:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
    def forward(self, input_ids=..., attention_mask=..., position_ids=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any:
        r"""
        Returns:

        Examples::

            >>> from transformers import CLIPTokenizer, CLIPTextModel

            >>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

            >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"],  padding=True, return_tensors="pt")

            >>> outputs = model(**inputs)
            >>> last_hidden_state = outputs.last_hidden_state
            >>> pooled_output = outputs.pooled_output # pooled (EOS token) states
        """
        ...
    


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def forward(self, pixel_values=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | BaseModelOutputWithPooling:
        r"""
        Returns:

        """
        ...
    


class CLIPVisionModel(CLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    def __init__(self, config: CLIPVisionConfig) -> None:
        ...
    
    def get_input_embeddings(self) -> nn.Module:
        ...
    
    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def forward(self, pixel_values=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any:
        r"""
        Returns:

        Examples::

            >>> from PIL import Image
            >>> import requests
            >>> from transformers import CLIPProcessor, CLIPVisionModel

            >>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> inputs = processor(images=image, return_tensors="pt")

            >>> outputs = model(**inputs)
            >>> last_hidden_state = outputs.last_hidden_state
            >>> pooled_output = outputs.pooled_output # pooled CLS states
        """
        ...
    


@add_start_docstrings(CLIP_START_DOCSTRING)
class CLIPModel(CLIPPreTrainedModel):
    config_class = CLIPConfig
    def __init__(self, config: CLIPConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(self, input_ids=..., attention_mask=..., position_ids=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any:
        r"""
        Returns:
            text_features (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, output_dim`): The text embeddings
            obtained by applying the projection layer to the pooled output of :class:`~transformers.CLIPTextModel`.

        Examples::

            >>> from transformers import CLIPTokenizer, CLIPModel

            >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

            >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"],  padding=True, return_tensors="pt")
            >>> text_features = model.get_text_features(**inputs)
        """
        ...
    
    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(self, pixel_values=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any:
        r"""
        Returns:
            image_features (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, output_dim`): The image embeddings
            obtained by applying the projection layer to the pooled output of :class:`~transformers.CLIPVisionModel`.

        Examples::

            >>> from PIL import Image
            >>> import requests
            >>> from transformers import CLIPProcessor, CLIPModel

            >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> inputs = processor(images=image, return_tensors="pt")

            >>> image_features = model.get_image_features(**inputs)
        """
        ...
    
    @add_start_docstrings_to_model_forward(CLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPOutput, config_class=CLIPConfig)
    def forward(self, input_ids=..., pixel_values=..., attention_mask=..., position_ids=..., return_loss=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> tuple[Tensor, Tensor, Tensor, Any, Any, Any, Any] | tuple[Tensor, Tensor, Any, Any, Any, Any] | CLIPOutput:
        r"""
        Returns:

        Examples::

            >>> from PIL import Image
            >>> import requests
            >>> from transformers import CLIPProcessor, CLIPModel

            >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

            >>> outputs = model(**inputs)
            >>> logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            >>> probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

        """
        ...
    


