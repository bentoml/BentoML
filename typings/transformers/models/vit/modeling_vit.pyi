

from torch import nn

from ...file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import BaseModelOutputWithPooling, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from .configuration_vit import ViTConfig

""" PyTorch ViT model. """
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
VIT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
def to_2tuple(x): # -> Iterable[Unknown] | tuple[Unknown, Unknown]:
    ...

class ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, pixel_values): # -> Any:
        ...
    


class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.

    """
    def __init__(self, image_size=..., patch_size=..., num_channels=..., embed_dim=...) -> None:
        ...
    
    def forward(self, pixel_values): # -> Any:
        ...
    


class ViTSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def transpose_for_scores(self, x):
        ...
    
    def forward(self, hidden_states, head_mask=..., output_attentions=...): # -> tuple[Tensor, Unknown | Any] | tuple[Tensor]:
        ...
    


class ViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor): # -> Any:
        ...
    


class ViTAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def forward(self, hidden_states, head_mask=..., output_attentions=...): # -> Any:
        ...
    


class ViTIntermediate(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any | Tensor:
        ...
    


class ViTOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor):
        ...
    


class ViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, head_mask=..., output_attentions=...): # -> Any:
        ...
    
    def feed_forward_chunk(self, attention_output): # -> Any:
        ...
    


class ViTEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, head_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


class ViTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ViTConfig
    base_model_prefix = ...


VIT_START_DOCSTRING = ...
VIT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare ViT Model transformer outputting raw hidden-states without any specific head on top.", VIT_START_DOCSTRING)
class ViTModel(ViTPreTrainedModel):
    def __init__(self, config, add_pooling_layer=...) -> None:
        ...
    
    def get_input_embeddings(self): # -> PatchEmbeddings:
        ...
    
    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values=..., head_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | BaseModelOutputWithPooling:
        r"""
        Returns:

        Examples::

            >>> from transformers import ViTFeatureExtractor, ViTModel
            >>> from PIL import Image
            >>> import requests

            >>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
            >>> model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

            >>> inputs = feature_extractor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        ...
    


class ViTPooler(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


@add_start_docstrings("""
    ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """, VIT_START_DOCSTRING)
class ViTForImageClassification(ViTPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values=..., head_mask=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | SequenceClassifierOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the image classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples::

            >>> from transformers import ViTFeatureExtractor, ViTForImageClassification
            >>> from PIL import Image
            >>> import requests

            >>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
            >>> model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

            >>> inputs = feature_extractor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> logits = outputs.logits
            >>> # model predicts one of the 1000 ImageNet classes
            >>> predicted_class_idx = logits.argmax(-1).item()
            >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
        """
        ...
    


