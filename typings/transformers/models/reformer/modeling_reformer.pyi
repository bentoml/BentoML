

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.autograd.function import Function

from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_outputs import (
    CausalLMOutput,
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_reformer import ReformerConfig

"""PyTorch REFORMER model. """
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = ...
LSHSelfAttentionOutput = ...
LocalSelfAttentionOutput = ...
AttentionOutput = ...
ReformerOutput = ...
ReformerBackwardOutput = ...
ReformerEncoderOutput = ...
class AxialPositionEmbeddings(nn.Module):
    """
    Constructs axial position embeddings. Useful for very long input sequences to save memory and time.
    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, position_ids): # -> Tensor:
        ...
    


class PositionEmbeddings(nn.Module):
    """Constructs conventional position embeddings of shape `[max_pos_embeddings, hidden_size]`."""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, position_ids):
        ...
    


class ReformerEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, input_ids=..., position_ids=..., inputs_embeds=..., start_idx_pos_encodings=...):
        ...
    


class EfficientAttentionMixin:
    """
    A few utilities for nn.Modules in Reformer, to be used as a mixin.
    """
    ...


class LSHSelfAttention(nn.Module, EfficientAttentionMixin):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., num_hashes=..., buckets=..., past_buckets_states=..., use_cache=..., output_attentions=..., **kwargs):
        ...
    


class ReverseSort(Function):
    """
    After chunked attention is applied which sorted clusters, original ordering has to be restored. Since customized
    backward function is used for Reformer, the gradients of the output vectors have to be explicitly sorted here.
    """
    @staticmethod
    def forward(ctx, out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx): # -> tuple[Tensor, Tensor]:
        ...
    
    @staticmethod
    def backward(ctx, grad_out_vectors, grad_logits): # -> tuple[Tensor, Tensor, None, None]:
        ...
    


class LocalSelfAttention(nn.Module, EfficientAttentionMixin):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., past_buckets_states=..., use_cache=..., output_attentions=..., **kwargs):
        ...
    


class ReformerSelfOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states):
        ...
    


class ReformerAttention(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., num_hashes=..., past_buckets_states=..., use_cache=..., orig_sequence_length=..., output_attentions=..., buckets=...): # -> AttentionOutput:
        ...
    


class ReformerFeedForwardDense(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Tensor:
        ...
    


class ReformerFeedForwardOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states):
        ...
    


class ChunkReformerFeedForward(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, attention_output): # -> Tensor:
        ...
    
    def forward_chunk(self, hidden_states): # -> Any:
        ...
    


class ReformerLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...
    
    def forward(self, prev_attn_output, hidden_states, attention_mask=..., head_mask=..., num_hashes=..., past_buckets_states=..., use_cache=..., orig_sequence_length=..., output_attentions=...): # -> ReformerOutput:
        ...
    
    def backward_pass(self, next_attn_output, hidden_states, grad_attn_output, grad_hidden_states, attention_mask=..., head_mask=..., buckets=...): # -> ReformerBackwardOutput:
        ...
    


class _ReversibleFunction(Function):
    """
    To prevent PyTorch from performing the usual backpropagation, a customized backward function is implemented here.
    This way it is made sure that no memory expensive activations are saved during the forward pass. This function is
    heavily inspired by https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py
    """
    @staticmethod
    def forward(ctx, hidden_states, layers, attention_mask, head_mask, num_hashes, all_hidden_states, all_attentions, past_buckets_states, use_cache, orig_sequence_length, output_hidden_states, output_attentions): # -> Tensor:
        ...
    
    @staticmethod
    def backward(ctx, grad_hidden_states): # -> tuple[Tensor, None, None, None, None, None, None, None, None, None, None, None]:
        ...
    


class ReformerEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., num_hashes=..., past_buckets_states=..., use_cache=..., orig_sequence_length=..., output_hidden_states=..., output_attentions=...): # -> ReformerEncoderOutput:
        ...
    


class ReformerOnlyLMHead(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Tensor:
        ...
    
    def forward_chunk(self, hidden_states): # -> Any:
        ...
    


class ReformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ReformerConfig
    base_model_prefix = ...
    @property
    def dummy_inputs(self): # -> dict[str, Tensor]:
        ...
    


@dataclass
class ReformerModelOutput(ModelOutput):
    """
    Output type of :class:`~transformers.ReformerModel`.

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_predict, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.

            ``num_predict`` corresponds to ``target_mapping.shape[1]``. If ``target_mapping`` is ``None``, then
            ``num_predict`` corresponds to ``sequence_length``.
        past_buckets_states (:obj:`List[Tuple(torch.LongTensor, torch.FloatTensor)]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`Tuple(torch.LongTensor, torch.FloatTensor` of length :obj:`config.n_layers`, with the first
            element being the previous `buckets` of shape :obj:`(batch_size, num_heads, num_hashes, sequence_length)`)
            and the second being the previous `hidden_states` of shape :obj:`(batch_size, sequence_length,
            hidden_size)`).

            Contains precomputed buckets and hidden-states that can be used (see ``past_buckets_states`` input) to
            speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings and one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    last_hidden_state: torch.FloatTensor
    past_buckets_states: Optional[List[Tuple[torch.LongTensor, torch.FloatTensor]]] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class ReformerModelWithLMHeadOutput(ModelOutput):
    """
    Output type of :class:`~transformers.ReformerModelWithLMHead`.

    Args:
        loss (:obj:`torch.FloatTensor` of shape `(1,)`, `optional`, returned when ``labels`` is provided)
            Language modeling loss (for next-token prediction).
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_predict, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

            ``num_predict`` corresponds to ``target_mapping.shape[1]``. If ``target_mapping`` is ``None``, then
            ``num_predict`` corresponds to ``sequence_length``.
        past_buckets_states (:obj:`List[Tuple(torch.LongTensor, torch.FloatTensor)]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`Tuple(torch.LongTensor, torch.FloatTensor` of length :obj:`config.n_layers`, with the first
            element being the previous `buckets` of shape :obj:`(batch_size, num_heads, num_hashes, sequence_length)`)
            and the second being the previous `hidden_states` of shape :obj:`(batch_size, sequence_length,
            hidden_size)`).

            Contains precomputed buckets and hidden-states that can be used (see ``past_buckets_states`` input) to
            speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            TTuple of :obj:`torch.FloatTensor` (one for the output of the embeddings and one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = ...
    logits: torch.FloatTensor = ...
    past_buckets_states: Optional[List[Tuple[torch.LongTensor, torch.FloatTensor]]] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


REFORMER_START_DOCSTRING = ...
REFORMER_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare Reformer Model transformer outputting raw hidden-states" "without any specific head on top.", REFORMER_START_DOCSTRING)
class ReformerModel(ReformerPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(REFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=ReformerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., position_ids=..., head_mask=..., inputs_embeds=..., num_hashes=..., past_buckets_states=..., use_cache=..., output_hidden_states=..., output_attentions=..., return_dict=...):
        ...
    


@add_start_docstrings("""Reformer Model with a `language modeling` head on top. """, REFORMER_START_DOCSTRING)
class ReformerModelWithLMHead(ReformerPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(REFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., position_ids=..., attention_mask=..., head_mask=..., inputs_embeds=..., num_hashes=..., past_buckets_states=..., use_cache=..., output_hidden_states=..., output_attentions=..., return_dict=..., labels=...): # -> Any | ReformerModelWithLMHeadOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0,
                ..., config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only
                computed for labels in ``[0, ..., config.vocab_size]``
        """
        ...
    
    def prepare_inputs_for_generation(self, input_ids, past=..., use_cache=..., num_hashes=..., **kwargs): # -> dict[str, Unknown]:
        ...
    


@add_start_docstrings("""Reformer Model with a `language modeling` head on top. """, REFORMER_START_DOCSTRING)
class ReformerForMaskedLM(ReformerPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(REFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., position_ids=..., attention_mask=..., head_mask=..., inputs_embeds=..., num_hashes=..., labels=..., output_hidden_states=..., output_attentions=..., return_dict=...): # -> Any | MaskedLMOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
                config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
                (masked), the loss is only computed for the tokens with labels
        """
        ...
    


@add_start_docstrings("""
    Reformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """, REFORMER_START_DOCSTRING)
class ReformerForSequenceClassification(ReformerPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(REFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., position_ids=..., attention_mask=..., head_mask=..., inputs_embeds=..., num_hashes=..., labels=..., output_hidden_states=..., output_attentions=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


class ReformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, **kwargs): # -> Any:
        ...
    


@add_start_docstrings("""
    Reformer Model with a span classification head on top for extractive question-answering tasks like SQuAD / TriviaQA
    ( a linear layer on top of hidden-states output to compute `span start logits` and `span end logits`.
    """, REFORMER_START_DOCSTRING)
class ReformerForQuestionAnswering(ReformerPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(REFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., position_ids=..., attention_mask=..., head_mask=..., inputs_embeds=..., num_hashes=..., start_positions=..., end_positions=..., output_hidden_states=..., output_attentions=..., return_dict=...):
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
    


