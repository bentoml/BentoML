

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import BaseModelOutputWithPooling, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from .configuration_deit import DeiTConfig

""" PyTorch DeiT model. """
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
DEIT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
def to_2tuple(x): # -> Iterable[Unknown] | tuple[Unknown, Unknown]:
    ...

class DeiTEmbeddings(nn.Module):
    """
    Construct the CLS token, distillation token, position and patch embeddings.

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
    


class DeiTSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def transpose_for_scores(self, x):
        ...
    
    def forward(self, hidden_states, head_mask=..., output_attentions=...): # -> tuple[Tensor, Unknown | Any] | tuple[Tensor]:
        ...
    


class DeiTSelfOutput(nn.Module):
    """
    The residual connection is defined in DeiTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor): # -> Any:
        ...
    


class DeiTAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def forward(self, hidden_states, head_mask=..., output_attentions=...): # -> Any:
        ...
    


class DeiTIntermediate(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any | Tensor:
        ...
    


class DeiTOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor):
        ...
    


class DeiTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, head_mask=..., output_attentions=...): # -> Any:
        ...
    
    def feed_forward_chunk(self, attention_output): # -> Any:
        ...
    


class DeiTEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, head_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


class DeiTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DeiTConfig
    base_model_prefix = ...


DEIT_START_DOCSTRING = ...
DEIT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare DeiT Model transformer outputting raw hidden-states without any specific head on top.", DEIT_START_DOCSTRING)
class DeiTModel(DeiTPreTrainedModel):
    def __init__(self, config, add_pooling_layer=...) -> None:
        ...
    
    def get_input_embeddings(self): # -> PatchEmbeddings:
        ...
    
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values=..., head_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | BaseModelOutputWithPooling:
        r"""
        Returns:

        Examples::

            >>> from transformers import DeiTFeatureExtractor, DeiTModel
            >>> from PIL import Image
            >>> import requests

            >>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> feature_extractor = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224')
            >>> model = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-224', add_pooling_layer=False)

            >>> inputs = feature_extractor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        ...
    


class DeiTPooler(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


@add_start_docstrings("""
    DeiT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """, DEIT_START_DOCSTRING)
class DeiTForImageClassification(DeiTPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values=..., head_mask=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | SequenceClassifierOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the image classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples::

            >>> from transformers import DeiTFeatureExtractor, DeiTForImageClassification
            >>> from PIL import Image
            >>> import requests

            >>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> # note: we are loading a DeiTForImageClassificationWithTeacher from the hub here,
            >>> # so the head will be randomly initialized, hence the predictions will be random
            >>> feature_extractor = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224')
            >>> model = DeiTForImageClassification.from_pretrained('facebook/deit-base-distilled-patch16-224')

            >>> inputs = feature_extractor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> logits = outputs.logits
            >>> # model predicts one of the 1000 ImageNet classes
            >>> predicted_class_idx = logits.argmax(-1).item()
            >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
        """
        ...
    


@dataclass
class DeiTForImageClassificationWithTeacherOutput(ModelOutput):
    """
    Output type of :class:`~transformers.DeiTForImageClassificationWithTeacher`.

    Args:
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Prediction scores as the average of the cls_logits and distillation logits.
        cls_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Prediction scores of the classification head (i.e. the linear layer on top of the final hidden state of the
            class token).
        distillation_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Prediction scores of the distillation head (i.e. the linear layer on top of the final hidden state of the
            distillation token).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of
            each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """
    logits: torch.FloatTensor = ...
    cls_logits: torch.FloatTensor = ...
    distillation_logits: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


@add_start_docstrings("""
    DeiT Model transformer with image classification heads on top (a linear layer on top of the final hidden state of
    the [CLS] token and a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet.

    .. warning::

           This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet
           supported.
    """, DEIT_START_DOCSTRING)
class DeiTForImageClassificationWithTeacher(DeiTPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=DeiTForImageClassificationWithTeacherOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values=..., head_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | DeiTForImageClassificationWithTeacherOutput:
        """
        Returns:

        Examples::

            >>> from transformers import DeiTFeatureExtractor, DeiTForImageClassificationWithTeacher
            >>> from PIL import Image
            >>> import requests

            >>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> feature_extractor = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224')
            >>> model = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-base-distilled-patch16-224')

            >>> inputs = feature_extractor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> logits = outputs.logits
            >>> # model predicts one of the 1000 ImageNet classes
            >>> predicted_class_idx = logits.argmax(-1).item()
            >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
        """
        ...
    


