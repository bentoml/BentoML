

import torch
from torch import nn

from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_outputs import (
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_deberta import DebertaConfig

""" PyTorch DeBERTa model. """
logger = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class ContextPooler(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any | Tensor:
        ...
    
    @property
    def output_dim(self):
        ...
    


class XSoftmax(torch.autograd.Function):
    """
    Masked Softmax which is optimized for saving memory

    Args:
        input (:obj:`torch.tensor`): The input tensor that will apply softmax.
        mask (:obj:`torch.IntTensor`): The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax

    Example::

          >>> import torch
          >>> from transformers.models.deberta.modeling_deberta import XSoftmax

          >>> # Make a tensor
          >>> x = torch.randn([4,20,100])

          >>> # Create a mask
          >>> mask = (x>0).int()

          >>> y = XSoftmax.apply(x, mask, dim=-1)
    """
    @staticmethod
    def forward(self, input, mask, dim): # -> Tensor:
        ...
    
    @staticmethod
    def backward(self, grad_output): # -> tuple[Tensor, None, None]:
        ...
    


class DropoutContext:
    def __init__(self) -> None:
        ...
    


def get_mask(input, local_context): # -> tuple[Unknown | None, Unknown | int]:
    ...

class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""
    @staticmethod
    def forward(ctx, input, local_ctx):
        ...
    
    @staticmethod
    def backward(ctx, grad_output): # -> tuple[Unknown, None]:
        ...
    


class StableDropout(nn.Module):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """
    def __init__(self, drop_prob) -> None:
        ...
    
    def forward(self, x):
        """
        Call the module

        Args:
            x (:obj:`torch.tensor`): The input tensor to apply dropout
        """
        ...
    
    def clear_context(self): # -> None:
        ...
    
    def init_context(self, reuse_mask=..., scale=...): # -> None:
        ...
    
    def get_context(self): # -> Unknown:
        ...
    


class DebertaLayerNorm(nn.Module):
    """LayerNorm module in the TF style (epsilon inside the square root)."""
    def __init__(self, size, eps=...) -> None:
        ...
    
    def forward(self, hidden_states):
        ...
    


class DebertaSelfOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor): # -> Any:
        ...
    


class DebertaAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask, return_att=..., query_states=..., relative_pos=..., rel_embeddings=...): # -> tuple[Any, Any | Unbound] | Any:
        ...
    


class DebertaIntermediate(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any | Tensor:
        ...
    


class DebertaOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor): # -> Any:
        ...
    


class DebertaLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask, return_att=..., query_states=..., relative_pos=..., rel_embeddings=...): # -> tuple[Any, Any | Unbound] | Any:
        ...
    


class DebertaEncoder(nn.Module):
    """Modified BertEncoder with relative position bias support"""
    def __init__(self, config) -> None:
        ...
    
    def get_rel_embedding(self): # -> Tensor | None:
        ...
    
    def get_attention_mask(self, attention_mask):
        ...
    
    def get_rel_pos(self, hidden_states, query_states=..., relative_pos=...): # -> Tensor:
        ...
    
    def forward(self, hidden_states, attention_mask, output_hidden_states=..., output_attentions=..., query_states=..., relative_pos=..., return_dict=...):
        ...
    


def build_relative_position(query_size, key_size, device): # -> Tensor:
    """
    Build relative position according to the query and key

    We assume the absolute position of query :math:`P_q` is range from (0, query_size) and the absolute position of key
    :math:`P_k` is range from (0, key_size), The relative positions from query to key is :math:`R_{q \\rightarrow k} =
    P_q - P_k`

    Args:
        query_size (int): the length of query
        key_size (int): the length of key

    Return:
        :obj:`torch.LongTensor`: A tensor with shape [1, query_size, key_size]

    """
    ...

@torch.jit.script
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    ...

@torch.jit.script
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    ...

@torch.jit.script
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    ...

class DisentangledSelfAttention(nn.Module):
    """
    Disentangled self-attention module

    Parameters:
        config (:obj:`str`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            `BertConfig`, for more details, please refer :class:`~transformers.DebertaConfig`

    """
    def __init__(self, config) -> None:
        ...
    
    def transpose_for_scores(self, x):
        ...
    
    def forward(self, hidden_states, attention_mask, return_att=..., query_states=..., relative_pos=..., rel_embeddings=...): # -> tuple[Tensor, Any] | Tensor:
        """
        Call the module

        Args:
            hidden_states (:obj:`torch.FloatTensor`):
                Input states to the module usually the output from previous layer, it will be the Q,K and V in
                `Attention(Q,K,V)`

            attention_mask (:obj:`torch.ByteTensor`):
                An attention mask matrix of shape [`B`, `N`, `N`] where `B` is the batch size, `N` is the maximum
                sequence length in which element [i,j] = `1` means the `i` th token in the input can attend to the `j`
                th token.

            return_att (:obj:`bool`, optional):
                Whether return the attention matrix.

            query_states (:obj:`torch.FloatTensor`, optional):
                The `Q` state in `Attention(Q,K,V)`.

            relative_pos (:obj:`torch.LongTensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [`B`, `N`, `N`] with
                values ranging in [`-max_relative_positions`, `max_relative_positions`].

            rel_embeddings (:obj:`torch.FloatTensor`):
                The embedding of relative distances. It's a tensor of shape [:math:`2 \\times
                \\text{max_relative_positions}`, `hidden_size`].


        """
        ...
    
    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        ...
    


class DebertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, input_ids=..., token_type_ids=..., position_ids=..., mask=..., inputs_embeds=...):
        ...
    


class DebertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DebertaConfig
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...
    _keys_to_ignore_on_load_unexpected = ...


DEBERTA_START_DOCSTRING = ...
DEBERTA_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare DeBERTa Model transformer outputting raw hidden-states without any specific head on top.", DEBERTA_START_DOCSTRING)
class DebertaModel(DebertaPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


@add_start_docstrings("""DeBERTa Model with a `language modeling` head on top. """, DEBERTA_START_DOCSTRING)
class DebertaForMaskedLM(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | MaskedLMOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        ...
    


class DebertaPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class DebertaLMPredictionHead(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class DebertaOnlyMLMHead(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, sequence_output): # -> Any:
        ...
    


@add_start_docstrings("""
    DeBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """, DEBERTA_START_DOCSTRING)
class DebertaForSequenceClassification(DebertaPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


@add_start_docstrings("""
    DeBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, DEBERTA_START_DOCSTRING)
class DebertaForTokenClassification(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | TokenClassifierOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        ...
    


@add_start_docstrings("""
    DeBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, DEBERTA_START_DOCSTRING)
class DebertaForQuestionAnswering(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., inputs_embeds=..., start_positions=..., end_positions=..., output_attentions=..., output_hidden_states=..., return_dict=...):
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
    


