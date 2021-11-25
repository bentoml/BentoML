

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
from .configuration_funnel import FunnelConfig

""" PyTorch Funnel Transformer model. """
logger = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST = ...
INF = ...
def load_tf_weights_in_funnel(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    ...

class FunnelEmbeddings(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, input_ids=..., inputs_embeds=...): # -> Any:
        ...
    


class FunnelAttentionStructure(nn.Module):
    """
    Contains helpers for `FunnelRelMultiheadAttention `.
    """
    cls_token_type_id: int = ...
    def __init__(self, config) -> None:
        ...
    
    def init_attention_inputs(self, inputs_embeds, attention_mask=..., token_type_ids=...): # -> tuple[tuple[Tensor, Tensor, Tensor, Tensor] | list[Unknown], Unknown | None, Unknown, Unknown | None]:
        """Returns the attention inputs associated to the inputs of the model."""
        ...
    
    def token_type_ids_to_mat(self, token_type_ids):
        """Convert `token_type_ids` to `token_type_mat`."""
        ...
    
    def get_position_embeds(self, seq_len, dtype, device): # -> tuple[Tensor, Tensor, Tensor, Tensor] | list[Unknown]:
        """
        Create and cache inputs related to relative position encoding. Those are very different depending on whether we
        are using the factorized or the relative shift attention:

        For the factorized attention, it returns the matrices (phi, pi, psi, omega) used in the paper, appendix A.2.2,
        final formula.

        For the relative shift attention, it returns all possible vectors R used in the paper, appendix A.2.1, final
        formula.

        Paper link: https://arxiv.org/abs/2006.03236
        """
        ...
    
    def stride_pool_pos(self, pos_id, block_index): # -> Tensor:
        """
        Pool `pos_id` while keeping the cls token separate (if `config.separate_cls=True`).
        """
        ...
    
    def relative_pos(self, pos, stride, pooled_pos=..., shift=...): # -> Tensor:
        """
        Build the relative positional vector between `pos` and `pooled_pos`.
        """
        ...
    
    def stride_pool(self, tensor, axis): # -> Any | None:
        """
        Perform pooling by stride slicing the tensor along the given axis.
        """
        ...
    
    def pool_tensor(self, tensor, mode=..., stride=...):
        """Apply 1D pooling to a tensor of size [B x T (x H)]."""
        ...
    
    def pre_attention_pooling(self, output, attention_inputs): # -> tuple[Any | Unknown | None, tuple[Unknown | Any | None, Unknown | Any | None, Unknown | Any | None, Unknown | Any | None]]:
        """Pool `output` and the proper parts of `attention_inputs` before the attention layer."""
        ...
    
    def post_attention_pooling(self, attention_inputs): # -> tuple[Unknown, Unknown | Any | None, Any | Unknown | None, Unknown | Any | None]:
        """Pool the proper parts of `attention_inputs` after the attention layer."""
        ...
    


class FunnelRelMultiheadAttention(nn.Module):
    def __init__(self, config, block_index) -> None:
        ...
    
    def relative_positional_attention(self, position_embeds, q_head, context_len, cls_mask=...): # -> Tensor:
        """Relative attention score for the positional encodings"""
        ...
    
    def relative_token_type_attention(self, token_type_mat, q_head, cls_mask=...): # -> Tensor | Literal[0]:
        """Relative attention score for the token_type_ids"""
        ...
    
    def forward(self, query, key, value, attention_inputs, output_attentions=...): # -> tuple[Any, Any] | tuple[Any]:
        ...
    


class FunnelPositionwiseFFN(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden): # -> Any:
        ...
    


class FunnelLayer(nn.Module):
    def __init__(self, config, block_index) -> None:
        ...
    
    def forward(self, query, key, value, attention_inputs, output_attentions=...): # -> tuple[Any, Any] | tuple[Any]:
        ...
    


class FunnelEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, inputs_embeds, attention_mask=..., token_type_ids=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


def upsample(x, stride, target_len, separate_cls=..., truncate_seq=...): # -> Tensor:
    """
    Upsample tensor `x` to match `target_len` by repeating the tokens `stride` time on the sequence length dimension.
    """
    ...

class FunnelDecoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, final_hidden, first_block_hidden, attention_mask=..., token_type_ids=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


class FunnelDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, discriminator_hidden_states): # -> Any:
        ...
    


class FunnelPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = FunnelConfig
    load_tf_weights = ...
    base_model_prefix = ...


class FunnelClassificationHead(nn.Module):
    def __init__(self, config, n_labels) -> None:
        ...
    
    def forward(self, hidden): # -> Any:
        ...
    


@dataclass
class FunnelForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.FunnelForPreTraining`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss of the ELECTRA-style objective.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
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
    logits: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


FUNNEL_START_DOCSTRING = ...
FUNNEL_INPUTS_DOCSTRING = ...
@add_start_docstrings("""
    The base Funnel Transformer Model transformer outputting raw hidden-states without upsampling head (also called
    decoder) or any task-specific head on top.
    """, FUNNEL_START_DOCSTRING)
class FunnelBaseModel(FunnelPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="funnel-transformer/small-base", output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


@add_start_docstrings("The bare Funnel Transformer Model transformer outputting raw hidden-states without any specific head on top.", FUNNEL_START_DOCSTRING)
class FunnelModel(FunnelPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


class FunnelForPreTraining(FunnelPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=FunnelForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | FunnelForPreTrainingOutput:
        r"""
        labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the ELECTRA-style loss. Input should be a sequence of tokens (see :obj:`input_ids`
            docstring) Indices should be in ``[0, 1]``:

            - 0 indicates the token is an original token,
            - 1 indicates the token was replaced.

        Returns:

        Examples::

            >>> from transformers import FunnelTokenizer, FunnelForPreTraining
            >>> import torch

            >>> tokenizer = FunnelTokenizer.from_pretrained('funnel-transformer/small')
            >>> model = FunnelForPreTraining.from_pretrained('funnel-transformer/small')

            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors= "pt")
            >>> logits = model(**inputs).logits
        """
        ...
    


@add_start_docstrings("""Funnel Transformer Model with a `language modeling` head on top. """, FUNNEL_START_DOCSTRING)
class FunnelForMaskedLM(FunnelPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC, mask="<mask>")
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | MaskedLMOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        ...
    


@add_start_docstrings("""
    Funnel Transformer Model with a sequence classification/regression head on top (two linear layer on top of the
    first timestep of the last hidden state) e.g. for GLUE tasks.
    """, FUNNEL_START_DOCSTRING)
class FunnelForSequenceClassification(FunnelPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="funnel-transformer/small-base", output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


@add_start_docstrings("""
    Funnel Transformer Model with a multiple choice classification head on top (two linear layer on top of the first
    timestep of the last hidden state, and a softmax) e.g. for RocStories/SWAG tasks.
    """, FUNNEL_START_DOCSTRING)
class FunnelForMultipleChoice(FunnelPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="funnel-transformer/small-base", output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        ...
    


@add_start_docstrings("""
    Funnel Transformer Model with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """, FUNNEL_START_DOCSTRING)
class FunnelForTokenClassification(FunnelPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | TokenClassifierOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        ...
    


@add_start_docstrings("""
    Funnel Transformer Model with a span classification head on top for extractive question-answering tasks like SQuAD
    (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, FUNNEL_START_DOCSTRING)
class FunnelForQuestionAnswering(FunnelPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., inputs_embeds=..., start_positions=..., end_positions=..., output_attentions=..., output_hidden_states=..., return_dict=...):
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
    


