

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_scipy_available,
    is_timm_available,
    is_vision_available,
    replace_return_docstrings,
)
from ...modeling_outputs import BaseModelOutputWithCrossAttentions, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from .configuration_detr import DetrConfig

""" PyTorch DETR model. """
if is_scipy_available():
    ...
if is_vision_available():
    ...
if is_timm_available():
    ...
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
DETR_PRETRAINED_MODEL_ARCHIVE_LIST = ...
@dataclass
class DetrDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    Base class for outputs of the DETR decoder. This class adds one attribute to BaseModelOutputWithCrossAttentions,
    namely an optional stack of intermediate decoder activations, i.e. the output of each decoder layer, each of them
    gone through a layernorm. This is useful when training the model with auxiliary decoding losses.

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of
            each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` and ``config.add_cross_attention=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the
            attention softmax, used to compute the weighted average in the cross-attention heads.
        intermediate_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(config.decoder_layers, batch_size, num_queries, hidden_size)`, `optional`, returned when ``config.auxiliary_loss=True``):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
    """
    intermediate_hidden_states: Optional[torch.FloatTensor] = ...


@dataclass
class DetrModelOutput(Seq2SeqModelOutput):
    """
    Base class for outputs of the DETR encoder-decoder model. This class adds one attribute to Seq2SeqModelOutput,
    namely an optional stack of intermediate decoder activations, i.e. the output of each decoder layer, each of them
    gone through a layernorm. This is useful when training the model with auxiliary decoding losses.

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`. Hidden-states of the decoder at the output of
            each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights of the decoder, after the attention softmax, used to
            compute the weighted average in the self-attention heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the
            attention softmax, used to compute the weighted average in the cross-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of
            each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights of the encoder, after the attention softmax, used to
            compute the weighted average in the self-attention heads.
        intermediate_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(config.decoder_layers, batch_size, sequence_length, hidden_size)`, `optional`, returned when ``config.auxiliary_loss=True``):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
    """
    intermediate_hidden_states: Optional[torch.FloatTensor] = ...


@dataclass
class DetrObjectDetectionOutput(ModelOutput):
    """
    Output type of :class:`~transformers.DetrForObjectDetection`.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (:obj:`Dict`, `optional`):
            A dictionary containing the individual losses. Useful for logging.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use :meth:`~transformers.DetrFeatureExtractor.post_process` to retrieve the
            unnormalized bounding boxes.
        auxiliary_outputs (:obj:`list[Dict]`, `optional`):
            Optional, only returned when auxilary losses are activated (i.e. :obj:`config.auxiliary_loss` is set to
            `True`) and labels are provided. It is a list of dictionnaries containing the two above keys (:obj:`logits`
            and :obj:`pred_boxes`) for each decoder layer.
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`. Hidden-states of the decoder at the output of
            each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights of the decoder, after the attention softmax, used to
            compute the weighted average in the self-attention heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the
            attention softmax, used to compute the weighted average in the cross-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of
            each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights of the encoder, after the attention softmax, used to
            compute the weighted average in the self-attention heads.
    """
    loss: Optional[torch.FloatTensor] = ...
    loss_dict: Optional[Dict] = ...
    logits: torch.FloatTensor = ...
    pred_boxes: torch.FloatTensor = ...
    auxiliary_outputs: Optional[List[Dict]] = ...
    last_hidden_state: Optional[torch.FloatTensor] = ...
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    encoder_last_hidden_state: Optional[torch.FloatTensor] = ...
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class DetrSegmentationOutput(ModelOutput):
    """
    Output type of :class:`~transformers.DetrForSegmentation`.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (:obj:`Dict`, `optional`):
            A dictionary containing the individual losses. Useful for logging.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use :meth:`~transformers.DetrFeatureExtractor.post_process` to retrieve the
            unnormalized bounding boxes.
        pred_masks (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_queries, height/4, width/4)`):
            Segmentation masks logits for all queries. See also
            :meth:`~transformers.DetrFeatureExtractor.post_process_segmentation` or
            :meth:`~transformers.DetrFeatureExtractor.post_process_panoptic` to evaluate instance and panoptic
            segmentation masks respectively.
        auxiliary_outputs (:obj:`list[Dict]`, `optional`):
            Optional, only returned when auxilary losses are activated (i.e. :obj:`config.auxiliary_loss` is set to
            `True`) and labels are provided. It is a list of dictionnaries containing the two above keys (:obj:`logits`
            and :obj:`pred_boxes`) for each decoder layer.
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`. Hidden-states of the decoder at the output of
            each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights of the decoder, after the attention softmax, used to
            compute the weighted average in the self-attention heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the
            attention softmax, used to compute the weighted average in the cross-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of
            each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights of the encoder, after the attention softmax, used to
            compute the weighted average in the self-attention heads.
    """
    loss: Optional[torch.FloatTensor] = ...
    loss_dict: Optional[Dict] = ...
    logits: torch.FloatTensor = ...
    pred_boxes: torch.FloatTensor = ...
    pred_masks: torch.FloatTensor = ...
    auxiliary_outputs: Optional[List[Dict]] = ...
    last_hidden_state: Optional[torch.FloatTensor] = ...
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    encoder_last_hidden_state: Optional[torch.FloatTensor] = ...
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = ...


class DetrFrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """
    def __init__(self, n) -> None:
        ...
    
    def forward(self, x):
        ...
    


def replace_batch_norm(m, name=...): # -> None:
    ...

class DetrTimmConvEncoder(nn.Module):
    """
    Convolutional encoder (backbone) from the timm library.

    nn.BatchNorm2d layers are replaced by DetrFrozenBatchNorm2d as defined above.

    """
    def __init__(self, name: str, dilation: bool) -> None:
        ...
    
    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor): # -> list[Unknown]:
        ...
    


class DetrConvModel(nn.Module):
    """
    This module adds 2D position embeddings to all intermediate feature maps of the convolutional encoder.
    """
    def __init__(self, conv_encoder, position_embedding) -> None:
        ...
    
    def forward(self, pixel_values, pixel_mask): # -> tuple[Unknown, list[Unknown]]:
        ...
    


class DetrSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """
    def __init__(self, embedding_dim=..., temperature=..., normalize=..., scale=...) -> None:
        ...
    
    def forward(self, pixel_values, pixel_mask): # -> Tensor:
        ...
    


class DetrLearnedPositionEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """
    def __init__(self, embedding_dim=...) -> None:
        ...
    
    def forward(self, pixel_values, pixel_mask=...): # -> Tensor:
        ...
    


def build_position_encoding(config): # -> DetrSinePositionEmbedding | DetrLearnedPositionEmbedding:
    ...

class DetrAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Here, we add position embeddings to the queries and keys (as explained in the DETR paper).
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = ..., is_decoder: bool = ..., bias: bool = ...) -> None:
        ...
    
    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]): # -> Tensor:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., position_embeddings: Optional[torch.Tensor] = ..., key_value_states: Optional[torch.Tensor] = ..., key_value_position_embeddings: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        ...
    


class DetrEncoderLayer(nn.Module):
    def __init__(self, config: DetrConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, position_embeddings: torch.Tensor = ..., output_attentions: bool = ...): # -> tuple[Tensor, Any] | tuple[Tensor]:
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape :obj:`(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                :obj:`(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_embeddings (:obj:`torch.FloatTensor`, `optional`): position embeddings, to be added to hidden_states.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        ...
    


class DetrDecoderLayer(nn.Module):
    def __init__(self, config: DetrConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., position_embeddings: Optional[torch.Tensor] = ..., query_position_embeddings: Optional[torch.Tensor] = ..., encoder_hidden_states: Optional[torch.Tensor] = ..., encoder_attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ...): # -> tuple[Tensor, Any, Any | None] | tuple[Tensor]:
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape :obj:`(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                :obj:`(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_embeddings (:obj:`torch.FloatTensor`, `optional`): position embeddings that are added to the queries and keys
            in the cross-attention layer.
            query_position_embeddings (:obj:`torch.FloatTensor`, `optional`): position embeddings that are added to the queries and keys
            in the self-attention layer.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape :obj:`(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                :obj:`(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        ...
    


class DetrClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor): # -> Tensor:
        ...
    


class DetrPreTrainedModel(PreTrainedModel):
    config_class = DetrConfig
    base_model_prefix = ...


DETR_START_DOCSTRING = ...
DETR_INPUTS_DOCSTRING = ...
class DetrEncoder(DetrPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`DetrEncoderLayer`.

    The encoder updates the flattened feature map through multiple self-attention layers.

    Small tweak for DETR:

    - position_embeddings are added to the forward pass.

    Args:
        config: DetrConfig
    """
    def __init__(self, config: DetrConfig) -> None:
        ...
    
    def forward(self, inputs_embeds=..., attention_mask=..., position_embeddings=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        Args:
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.

            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding pixel features. Mask values selected in ``[0, 1]``:

                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).

                `What are attention masks? <../glossary.html#attention-mask>`__

            position_embeddings (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.

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
    


class DetrDecoder(DetrPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`DetrDecoderLayer`.

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some small tweaks for DETR:

    - position_embeddings and query_position_embeddings are added to the forward pass.
    - if self.config.auxiliary_loss is set to True, also returns a stack of activations from all decoding layers.

    Args:
        config: DetrConfig
    """
    def __init__(self, config: DetrConfig) -> None:
        ...
    
    def forward(self, inputs_embeds=..., attention_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., position_embeddings=..., query_position_embeddings=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        Args:
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
                The query embeddings that are passed into the decoder.

            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on certain queries. Mask values selected in ``[0, 1]``:

                - 1 for queries that are **not masked**,
                - 0 for queries that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in ``[0, 1]``:

                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).

            position_embeddings (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Position embeddings that are added to the queries and keys in each cross-attention layer.
            query_position_embeddings (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_queries, hidden_size)`):, `optional`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
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
    


@add_start_docstrings("""
    The bare DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw hidden-states without
    any specific head on top.
    """, DETR_START_DOCSTRING)
class DetrModel(DetrPreTrainedModel):
    def __init__(self, config: DetrConfig) -> None:
        ...
    
    def get_encoder(self): # -> DetrEncoder:
        ...
    
    def get_decoder(self): # -> DetrDecoder:
        ...
    
    def freeze_backbone(self): # -> None:
        ...
    
    def unfreeze_backbone(self): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DetrModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values, pixel_mask=..., decoder_attention_mask=..., encoder_outputs=..., inputs_embeds=..., decoder_inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        Returns:

        Examples::

            >>> from transformers import DetrFeatureExtractor, DetrModel
            >>> from PIL import Image
            >>> import requests

            >>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
            >>> model = DetrModel.from_pretrained('facebook/detr-resnet-50')
            >>> inputs = feature_extractor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        ...
    


@add_start_docstrings("""
    DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on top, for tasks
    such as COCO detection.
    """, DETR_START_DOCSTRING)
class DetrForObjectDetection(DetrPreTrainedModel):
    def __init__(self, config: DetrConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DetrObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values, pixel_mask=..., decoder_attention_mask=..., encoder_outputs=..., inputs_embeds=..., decoder_inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`List[Dict]` of len :obj:`(batch_size,)`, `optional`):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a :obj:`torch.LongTensor` of len :obj:`(number of
            bounding boxes in the image,)` and the boxes a :obj:`torch.FloatTensor` of shape :obj:`(number of bounding
            boxes in the image, 4)`.

        Returns:

        Examples::

            >>> from transformers import DetrFeatureExtractor, DetrForObjectDetection
            >>> from PIL import Image
            >>> import requests

            >>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
            >>> model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

            >>> inputs = feature_extractor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> # model predicts bounding boxes and corresponding COCO classes
            >>> logits = outputs.logits
            >>> bboxes = outputs.pred_boxes
        """
        ...
    


@add_start_docstrings("""
    DETR Model (consisting of a backbone and encoder-decoder Transformer) with a segmentation head on top, for tasks
    such as COCO panoptic.

    """, DETR_START_DOCSTRING)
class DetrForSegmentation(DetrPreTrainedModel):
    def __init__(self, config: DetrConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DetrSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values, pixel_mask=..., decoder_attention_mask=..., encoder_outputs=..., inputs_embeds=..., decoder_inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`List[Dict]` of len :obj:`(batch_size,)`, `optional`):
            Labels for computing the bipartite matching loss, DICE/F-1 loss and Focal loss. List of dicts, each
            dictionary containing at least the following 3 keys: 'class_labels', 'boxes' and 'masks' (the class labels,
            bounding boxes and segmentation masks of an image in the batch respectively). The class labels themselves
            should be a :obj:`torch.LongTensor` of len :obj:`(number of bounding boxes in the image,)`, the boxes a
            :obj:`torch.FloatTensor` of shape :obj:`(number of bounding boxes in the image, 4)` and the masks a
            :obj:`torch.FloatTensor` of shape :obj:`(number of bounding boxes in the image, height, width)`.

        Returns:

        Examples::

            >>> from transformers import DetrFeatureExtractor, DetrForSegmentation
            >>> from PIL import Image
            >>> import requests

            >>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50-panoptic')
            >>> model = DetrForSegmentation.from_pretrained('facebook/detr-resnet-50-panoptic')

            >>> inputs = feature_extractor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> # model predicts COCO classes, bounding boxes, and masks
            >>> logits = outputs.logits
            >>> bboxes = outputs.pred_boxes
            >>> masks = outputs.pred_masks
        """
        ...
    


class DetrMaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm. Upsampling is done using a FPN approach
    """
    def __init__(self, dim, fpn_dims, context_dim) -> None:
        ...
    
    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]): # -> Tensor:
        ...
    


class DetrMHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""
    def __init__(self, query_dim, hidden_dim, num_heads, dropout=..., bias=..., std=...) -> None:
        ...
    
    def forward(self, q, k, mask: Optional[Tensor] = ...): # -> Any:
        ...
    


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
    """
    ...

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = ..., gamma: float = ...):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    ...

class DetrLoss(nn.Module):
    """
    This class computes the losses for DetrForObjectDetection/DetrForSegmentation. The process happens in two steps: 1)
    we compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair
    of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, matcher, num_classes, eos_coef, losses) -> None:
        """
        Create the criterion.

        A note on the num_classes parameter (copied from original repo in detr.py): "the naming of the `num_classes`
        parameter of the criterion is somewhat misleading. it indeed corresponds to `max_obj_id + 1`, where max_obj_id
        is the maximum id for a class in your dataset. For example, COCO has a max_obj_id of 90, so we pass
        `num_classes` to be 91. As another example, for a dataset that has a single class with id 1, you should pass
        `num_classes` to be 2 (max_obj_id + 1). For more details on this, check the following discussion
        https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223"

        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            num_classes: number of object categories, omitting the special no-object category.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        ...
    
    def loss_labels(self, outputs, targets, indices, num_boxes): # -> dict[str, Unknown]:
        """
        Classification loss (NLL) targets dicts must contain the key "class_labels" containing a tensor of dim
        [nb_target_boxes]
        """
        ...
    
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes): # -> dict[str, Unknown]:
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        ...
    
    def loss_boxes(self, outputs, targets, indices, num_boxes): # -> dict[Unknown, Unknown]:
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        ...
    
    def loss_masks(self, outputs, targets, indices, num_boxes): # -> dict[str, Unknown]:
        """
        Compute the losses related to the masks: the focal loss and the dice loss.

        Targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        ...
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes): # -> dict[str, Unknown]:
        ...
    
    def forward(self, outputs, targets): # -> dict[Unknown, Unknown]:
        """
        This performs the loss computation.

        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        ...
    


class DetrMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers) -> None:
        ...
    
    def forward(self, x): # -> Any:
        ...
    


class DetrHungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).
    """
    def __init__(self, class_cost: float = ..., bbox_cost: float = ..., giou_cost: float = ...) -> None:
        """
        Creates the matcher.

        Params:
            class_cost: This is the relative weight of the classification error in the matching cost
            bbox_cost: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            giou_cost: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        ...
    
    @torch.no_grad()
    def forward(self, outputs, targets): # -> list[tuple[Tensor, Tensor]]:
        """
        Performs the matching.

        Params:
            outputs: This is a dict that contains at least these entries:
                 "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                 objects in the target) containing the class labels "boxes": Tensor of dim [num_target_boxes, 4]
                 containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:

                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        ...
    


def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        area (Tensor[N]): area for each box
    """
    ...

def box_iou(boxes1, boxes2): # -> tuple[Tensor, Tensor]:
    ...

def generalized_box_iou(boxes1, boxes2): # -> Tensor:
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

    Returns:
        a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    ...

class NestedTensor:
    def __init__(self, tensors, mask: Optional[Tensor]) -> None:
        ...
    
    def to(self, device: Device) -> NestedTensor:
        ...
    
    def decompose(self): # -> tuple[Unknown, Tensor | None]:
        ...
    
    def __repr__(self): # -> str:
        ...
    


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]): # -> NestedTensor:
    ...

