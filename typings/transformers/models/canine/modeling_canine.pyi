

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_outputs import (
    ModelOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_canine import CanineConfig

""" PyTorch CANINE model. """
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
CANINE_PRETRAINED_MODEL_ARCHIVE_LIST = ...
_PRIMES = ...
@dataclass
class CanineModelOutputWithPooling(ModelOutput):
    """
    Output type of :class:`~transformers.CanineModel`. Based on
    :class:`~transformers.modeling_outputs.BaseModelOutputWithPooling`, but with slightly different
    :obj:`hidden_states` and :obj:`attentions`, as these also include the hidden states and attentions of the shallow
    Transformer encoders.

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model (i.e. the output of the final
            shallow Transformer encoder).
        pooler_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
            Hidden-state of the first token of the sequence (classification token) at the last layer of the deep
            Transformer encoder, further processed by a Linear layer and a Tanh activation function. The Linear layer
            weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the input to each encoder + one for the output of each layer of
            each encoder) of shape :obj:`(batch_size, sequence_length, hidden_size)` and :obj:`(batch_size,
            sequence_length // config.downsampling_rate, hidden_size)`. Hidden-states of the model at the output of
            each layer plus the initial input to each Transformer encoder. The hidden states of the shallow encoders
            have length :obj:`sequence_length`, but the hidden states of the deep encoder have length
            :obj:`sequence_length` // :obj:`config.downsampling_rate`.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of the 3 Transformer encoders of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)` and :obj:`(batch_size, num_heads,
            sequence_length // config.downsampling_rate, sequence_length // config.downsampling_rate)`. Attentions
            weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    """
    last_hidden_state: torch.FloatTensor = ...
    pooler_output: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


def load_tf_weights_in_canine(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    ...

class CanineEmbeddings(nn.Module):
    """Construct the character, position and token_type embeddings."""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, input_ids=..., token_type_ids=..., position_ids=..., inputs_embeds=...): # -> Any:
        ...
    


class CharactersToMolecules(nn.Module):
    """Convert character sequence to initial molecule sequence (i.e. downsample) using strided convolutions."""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, char_encoding: torch.Tensor) -> torch.Tensor:
        ...
    


class ConvProjection(nn.Module):
    """
    Project representations from hidden_size*2 back to hidden_size across a window of w = config.upsampling_kernel_size
    characters.
    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, inputs, final_seq_char_positions=...): # -> Any:
        ...
    


class CanineSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def transpose_for_scores(self, x):
        ...
    
    def forward(self, from_tensor, to_tensor, attention_mask=..., head_mask=..., output_attentions=...):
        ...
    


class CanineSelfOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor): # -> Any:
        ...
    


class CanineAttention(nn.Module):
    """
    Additional arguments related to local attention:

        - **local** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Whether to apply local attention.
        - **always_attend_to_first_position** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Should all blocks
          be able to attend
        to the :obj:`to_tensor`'s first position (e.g. a [CLS] position)? - **first_position_attends_to_all**
        (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Should the `from_tensor`'s first position be able to
        attend to all positions within the `from_tensor`? - **attend_from_chunk_width** (:obj:`int`, `optional`,
        defaults to 128) -- The width of each block-wise chunk in :obj:`from_tensor`. - **attend_from_chunk_stride**
        (:obj:`int`, `optional`, defaults to 128) -- The number of elements to skip when moving to the next block in
        :obj:`from_tensor`. - **attend_to_chunk_width** (:obj:`int`, `optional`, defaults to 128) -- The width of each
        block-wise chunk in `to_tensor`. - **attend_to_chunk_stride** (:obj:`int`, `optional`, defaults to 128) -- The
        number of elements to skip when moving to the next block in :obj:`to_tensor`.
    """
    def __init__(self, config, local=..., always_attend_to_first_position: bool = ..., first_position_attends_to_all: bool = ..., attend_from_chunk_width: int = ..., attend_from_chunk_stride: int = ..., attend_to_chunk_width: int = ..., attend_to_chunk_stride: int = ...) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., output_attentions=...):
        ...
    


class CanineIntermediate(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any | Tensor:
        ...
    


class CanineOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor): # -> Any:
        ...
    


class CanineLayer(nn.Module):
    def __init__(self, config, local, always_attend_to_first_position, first_position_attends_to_all, attend_from_chunk_width, attend_from_chunk_stride, attend_to_chunk_width, attend_to_chunk_stride) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., output_attentions=...): # -> Any:
        ...
    
    def feed_forward_chunk(self, attention_output): # -> Any:
        ...
    


class CanineEncoder(nn.Module):
    def __init__(self, config, local=..., always_attend_to_first_position=..., first_position_attends_to_all=..., attend_from_chunk_width=..., attend_from_chunk_stride=..., attend_to_chunk_width=..., attend_to_chunk_stride=...) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


class CaninePooler(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class CaninePredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class CanineLMPredictionHead(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class CanineOnlyMLMHead(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, sequence_output): # -> Any:
        ...
    


class CaninePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = CanineConfig
    load_tf_weights = ...
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...


CANINE_START_DOCSTRING = ...
CANINE_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare CANINE Model transformer outputting raw hidden-states without any specific head on top.", CANINE_START_DOCSTRING)
class CanineModel(CaninePreTrainedModel):
    def __init__(self, config, add_pooling_layer=...) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CANINE_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=CanineModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


@add_start_docstrings("""
    CANINE Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """, CANINE_START_DOCSTRING)
class CanineForSequenceClassification(CaninePreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CANINE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


@add_start_docstrings("""
    CANINE Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, CANINE_START_DOCSTRING)
class CanineForMultipleChoice(CaninePreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CANINE_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        ...
    


@add_start_docstrings("""
    CANINE Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, CANINE_START_DOCSTRING)
class CanineForTokenClassification(CaninePreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CANINE_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | TokenClassifierOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        ...
    


@add_start_docstrings("""
    CANINE Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, CANINE_START_DOCSTRING)
class CanineForQuestionAnswering(CaninePreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CANINE_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., start_positions=..., end_positions=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        ...
    


