

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
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_xlm import XLMConfig

"""
 PyTorch XLM model.
"""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
XLM_PRETRAINED_MODEL_ARCHIVE_LIST = ...
def create_sinusoidal_embeddings(n_pos, dim, out): # -> None:
    ...

def get_masks(slen, lengths, causal, padding_mask=...): # -> tuple[Unknown, Tensor | Unknown]:
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    ...

class MultiHeadAttention(nn.Module):
    NEW_ID = ...
    def __init__(self, n_heads, dim, config) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def forward(self, input, mask, kv=..., cache=..., head_mask=..., output_attentions=...):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        ...
    


class TransformerFFN(nn.Module):
    def __init__(self, in_dim, dim_hidden, out_dim, config) -> None:
        ...
    
    def forward(self, input): # -> Tensor:
        ...
    
    def ff_chunk(self, input):
        ...
    


class XLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = XLMConfig
    load_tf_weights = ...
    base_model_prefix = ...
    def __init__(self, *inputs, **kwargs) -> None:
        ...
    
    @property
    def dummy_inputs(self): # -> dict[str, Tensor | None]:
        ...
    


@dataclass
class XLMForQuestionAnsweringOutput(ModelOutput):
    """
    Base class for outputs of question answering models using a :obj:`SquadHead`.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned if both :obj:`start_positions` and :obj:`end_positions` are provided):
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification
            losses.
        start_top_log_probs (``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top)``, `optional`, returned if ``start_positions`` or ``end_positions`` is not provided):
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        start_top_index (``torch.LongTensor`` of shape ``(batch_size, config.start_n_top)``, `optional`, returned if ``start_positions`` or ``end_positions`` is not provided):
            Indices for the top config.start_n_top start token possibilities (beam-search).
        end_top_log_probs (``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``, `optional`, returned if ``start_positions`` or ``end_positions`` is not provided):
            Log probabilities for the top ``config.start_n_top * config.end_n_top`` end token possibilities
            (beam-search).
        end_top_index (``torch.LongTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``, `optional`, returned if ``start_positions`` or ``end_positions`` is not provided):
            Indices for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        cls_logits (``torch.FloatTensor`` of shape ``(batch_size,)``, `optional`, returned if ``start_positions`` or ``end_positions`` is not provided):
            Log probabilities for the ``is_impossible`` label of the answers.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = ...
    start_top_log_probs: Optional[torch.FloatTensor] = ...
    start_top_index: Optional[torch.LongTensor] = ...
    end_top_log_probs: Optional[torch.FloatTensor] = ...
    end_top_index: Optional[torch.LongTensor] = ...
    cls_logits: Optional[torch.FloatTensor] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


XLM_START_DOCSTRING = ...
XLM_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare XLM Model transformer outputting raw hidden-states without any specific head on top.", XLM_START_DOCSTRING)
class XLMModel(XLMPreTrainedModel):
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., langs=..., token_type_ids=..., position_ids=..., lengths=..., cache=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


class XLMPredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, x, y=...): # -> tuple[Unknown, Any] | tuple[Any] | tuple[Any, Unknown | Any | Tensor] | tuple[Unknown | Any | Tensor]:
        """Compute the loss, and optionally the scores."""
        ...
    


@add_start_docstrings("""
    The XLM Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """, XLM_START_DOCSTRING)
class XLMWithLMHeadModel(XLMPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear | AdaptiveLogSoftmaxWithLoss:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs): # -> dict[str, Tensor | None]:
        ...
    
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC, mask="<special1>")
    def forward(self, input_ids=..., attention_mask=..., langs=..., token_type_ids=..., position_ids=..., lengths=..., cache=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | MaskedLMOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        ...
    


@add_start_docstrings("""
    XLM Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g.
    for GLUE tasks.
    """, XLM_START_DOCSTRING)
class XLMForSequenceClassification(XLMPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., langs=..., token_type_ids=..., position_ids=..., lengths=..., cache=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


@add_start_docstrings("""
    XLM Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, XLM_START_DOCSTRING)
class XLMForQuestionAnsweringSimple(XLMPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., langs=..., token_type_ids=..., position_ids=..., lengths=..., cache=..., head_mask=..., inputs_embeds=..., start_positions=..., end_positions=..., output_attentions=..., output_hidden_states=..., return_dict=...):
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
    


@add_start_docstrings("""
    XLM Model with a beam-search span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, XLM_START_DOCSTRING)
class XLMForQuestionAnswering(XLMPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=XLMForQuestionAnsweringOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., langs=..., token_type_ids=..., position_ids=..., lengths=..., cache=..., head_mask=..., inputs_embeds=..., start_positions=..., end_positions=..., is_impossible=..., cls_index=..., p_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | XLMForQuestionAnsweringOutput:
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        is_impossible (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels whether a question has an answer or no answer (SQuAD 2.0)
        cls_index (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for position (index) of the classification token to use as input for computing plausibility of the
            answer.
        p_mask (``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Optional mask of tokens which can't be in answers (e.g. [CLS], [PAD], ...). 1.0 means token should be
            masked. 0.0 mean token is not masked.

        Returns:

        Example::

            >>> from transformers import XLMTokenizer, XLMForQuestionAnswering
            >>> import torch

            >>> tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
            >>> model = XLMForQuestionAnswering.from_pretrained('xlm-mlm-en-2048')

            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            >>> start_positions = torch.tensor([1])
            >>> end_positions = torch.tensor([3])

            >>> outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
            >>> loss = outputs.loss
        """
        ...
    


@add_start_docstrings("""
    XLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, XLM_START_DOCSTRING)
class XLMForTokenClassification(XLMPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., langs=..., token_type_ids=..., position_ids=..., lengths=..., cache=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | TokenClassifierOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        ...
    


@add_start_docstrings("""
    XLM Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, XLM_START_DOCSTRING)
class XLMForMultipleChoice(XLMPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., langs=..., token_type_ids=..., position_ids=..., lengths=..., cache=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        ...
    


