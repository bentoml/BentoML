

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_utils import PreTrainedModel
from .configuration_lxmert import LxmertConfig

""" PyTorch LXMERT model. """
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class GeLU(nn.Module):
    def __init__(self) -> None:
        ...
    
    def forward(self, x):
        ...
    


@dataclass
class LxmertModelOutput(ModelOutput):
    """
    Lxmert's outputs that contain the last hidden states, pooled outputs, and attention probabilities for the language,
    visual, and, cross-modality encoders. (note: the visual encoder in Lxmert is referred to as the "relation-ship"
    encoder")


    Args:
        language_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the language encoder.
        vision_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the visual encoder.
        pooled_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification, CLS, token) further processed
            by a Linear layer and a Tanh activation function. The Linear
        language_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        language_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        vision_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """
    language_output: Optional[torch.FloatTensor] = ...
    vision_output: Optional[torch.FloatTensor] = ...
    pooled_output: Optional[torch.FloatTensor] = ...
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    language_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class LxmertForQuestionAnsweringOutput(ModelOutput):
    """
    Output type of :class:`~transformers.LxmertForQuestionAnswering`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.k.
        question_answering_score: (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, n_qa_answers)`, `optional`):
            Prediction scores of question answering objective (classification).
        language_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        language_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        vision_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """
    loss: Optional[torch.FloatTensor] = ...
    question_answering_score: Optional[torch.FloatTensor] = ...
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    language_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class LxmertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.LxmertForPreTraining`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cross_relationship_score: (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the textual matching objective (classification) head (scores of True/False
            continuation before SoftMax).
        question_answering_score: (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, n_qa_answers)`):
            Prediction scores of question answering objective (classification).
        language_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        language_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        vision_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.

    """
    loss: [torch.FloatTensor] = ...
    prediction_logits: Optional[torch.FloatTensor] = ...
    cross_relationship_score: Optional[torch.FloatTensor] = ...
    question_answering_score: Optional[torch.FloatTensor] = ...
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    language_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = ...


def load_tf_weights_in_lxmert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    ...

class LxmertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, input_ids, token_type_ids=..., inputs_embeds=...): # -> Any:
        ...
    


class LxmertAttention(nn.Module):
    def __init__(self, config, ctx_dim=...) -> None:
        ...
    
    def transpose_for_scores(self, x):
        ...
    
    def forward(self, hidden_states, context, attention_mask=..., output_attentions=...): # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...
    


class LxmertAttentionOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor): # -> Any:
        ...
    


class LxmertCrossAttentionLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=..., output_attentions=...): # -> tuple[Any, Any | Unbound] | tuple[Any]:
        ...
    


class LxmertSelfAttentionLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, input_tensor, attention_mask, output_attentions=...): # -> tuple[Any, Any | Unbound] | tuple[Any]:
        ...
    


class LxmertIntermediate(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any | Tensor:
        ...
    


class LxmertOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor): # -> Any:
        ...
    


class LxmertLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., output_attentions=...): # -> Any:
        ...
    


class LxmertXLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def cross_att(self, lang_input, lang_attention_mask, visual_input, visual_attention_mask, output_x_attentions=...): # -> tuple[Any, Any]:
        ...
    
    def self_att(self, lang_input, lang_attention_mask, visual_input, visual_attention_mask): # -> tuple[Any, Any]:
        ...
    
    def output_fc(self, lang_input, visual_input): # -> tuple[Any, Any]:
        ...
    
    def forward(self, lang_feats, lang_attention_mask, visual_feats, visual_attention_mask, output_attentions=...): # -> tuple[Any, Any, Any] | tuple[Any, Any]:
        ...
    


class LxmertVisualFeatureEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, visual_feats, visual_pos): # -> Any:
        ...
    


class LxmertEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, lang_feats, lang_attention_mask, visual_feats, visual_pos, visual_attention_mask=..., output_attentions=...):
        ...
    


class LxmertPooler(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class LxmertPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class LxmertLMPredictionHead(nn.Module):
    def __init__(self, config, lxmert_model_embedding_weights) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class LxmertVisualAnswerHead(nn.Module):
    def __init__(self, config, num_labels) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class LxmertVisualObjHead(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> dict[Unknown, Unknown]:
        ...
    


class LxmertPreTrainingHeads(nn.Module):
    def __init__(self, config, lxmert_model_embedding_weights) -> None:
        ...
    
    def forward(self, sequence_output, pooled_output): # -> tuple[Any, Any]:
        ...
    


class LxmertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = LxmertConfig
    load_tf_weights = ...
    base_model_prefix = ...


LXMERT_START_DOCSTRING = ...
LXMERT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare Lxmert Model transformer outputting raw hidden-states without any specific head on top.", LXMERT_START_DOCSTRING)
class LxmertModel(LxmertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=LxmertModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., visual_feats=..., visual_pos=..., attention_mask=..., visual_attention_mask=..., token_type_ids=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


@add_start_docstrings("""Lxmert Model with a specified pretraining head on top. """, LXMERT_START_DOCSTRING)
class LxmertForPreTraining(LxmertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def resize_num_qa_labels(self, num_labels): # -> Module | None:
        """
        Build a resized question answering linear layer Module from a provided new linear layer. Increasing the size
        will add newly initialized weights. Reducing the size will remove weights from the end

        Args:
            num_labels (:obj:`int`, `optional`):
                New number of labels in the linear layer weight matrix. Increasing the size will add newly initialized
                weights at the end. Reducing the size will remove weights from the end. If not provided or :obj:`None`,
                just returns a pointer to the qa labels :obj:`torch.nn.Linear`` module of the model without doing
                anything.

        Return:
            :obj:`torch.nn.Linear`: Pointer to the resized Linear layer or the old Linear layer
        """
        ...
    
    def get_qa_logit_layer(self) -> nn.Module:
        """
        Returns the the linear layer that produces question answering logits.

        Returns:
            :obj:`nn.Module`: A torch module mapping the question answering prediction hidden states or :obj:`None` if
            LXMERT does not have a visual answering head.
        """
        ...
    
    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=LxmertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., visual_feats=..., visual_pos=..., attention_mask=..., visual_attention_mask=..., token_type_ids=..., inputs_embeds=..., labels=..., obj_labels=..., matched_label=..., ans=..., output_attentions=..., output_hidden_states=..., return_dict=..., **kwargs):
        r"""
        labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        obj_labels: (``Dict[Str: Tuple[Torch.FloatTensor, Torch.FloatTensor]]``, `optional`):
            each key is named after each one of the visual losses and each element of the tuple is of the shape
            ``(batch_size, num_features)`` and ``(batch_size, num_features, visual_feature_dim)`` for each the label id
            and the label score respectively
        matched_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the whether or not the text input matches the image (classification) loss. Input
            should be a sequence pair (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:

            - 0 indicates that the sentence does not match the image,
            - 1 indicates that the sentence does match the image.
        ans: (``Torch.Tensor`` of shape ``(batch_size)``, `optional`):
            a one hot representation hof the correct answer `optional`

        Returns:
        """
        ...
    


@add_start_docstrings("""Lxmert Model with a visual-answering head on top for downstream QA tasks""", LXMERT_START_DOCSTRING)
class LxmertForQuestionAnswering(LxmertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def resize_num_qa_labels(self, num_labels): # -> Module | None:
        """
        Build a resized question answering linear layer Module from a provided new linear layer. Increasing the size
        will add newly initialized weights. Reducing the size will remove weights from the end

        Args:
            num_labels (:obj:`int`, `optional`):
                New number of labels in the linear layer weight matrix. Increasing the size will add newly initialized
                weights at the end. Reducing the size will remove weights from the end. If not provided or :obj:`None`,
                just returns a pointer to the qa labels :obj:`torch.nn.Linear`` module of the model without doing
                anything.

        Return:
            :obj:`torch.nn.Linear`: Pointer to the resized Linear layer or the old Linear layer
        """
        ...
    
    def get_qa_logit_layer(self) -> nn.Module:
        """
        Returns the the linear layer that produces question answering logits

        Returns:
            :obj:`nn.Module`: A torch module mapping the question answering prediction hidden states. :obj:`None`: A
            NoneType object if Lxmert does not have the visual answering head.
        """
        ...
    
    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=LxmertForQuestionAnsweringOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., visual_feats=..., visual_pos=..., attention_mask=..., visual_attention_mask=..., token_type_ids=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | LxmertForQuestionAnsweringOutput:
        r"""
        labels: (``Torch.Tensor`` of shape ``(batch_size)``, `optional`):
            A one-hot representation of the correct answer

        Returns:
        """
        ...
    


