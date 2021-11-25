

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn

from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_utils import PreTrainedModel
from .configuration_transfo_xl import TransfoXLConfig

"""
 PyTorch Transformer XL model. Adapted from https://github.com/kimiyoung/transformer-xl. In particular
 https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
"""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST = ...
def build_tf_to_pytorch_map(model, config):
    """
    A map of modules from TF to PyTorch. This time I use a map to keep the PyTorch model as identical to the original
    PyTorch model as possible.
    """
    ...

def load_tf_weights_in_transfo_xl(model, config, tf_path):
    """Load tf checkpoints in a pytorch model"""
    ...

class PositionalEmbedding(nn.Module):
    def __init__(self, demb) -> None:
        ...
    
    def forward(self, pos_seq, bsz=...): # -> Tensor:
        ...
    


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=..., layer_norm_epsilon=...) -> None:
        ...
    
    def forward(self, inp): # -> Any:
        ...
    


class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=..., pre_lnorm=..., r_r_bias=..., r_w_bias=..., layer_norm_epsilon=...) -> None:
        ...
    
    def forward(self, w, r, attn_mask=..., mems=..., head_mask=..., output_attentions=...):
        ...
    


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, layer_norm_epsilon=..., **kwargs) -> None:
        ...
    
    def forward(self, dec_inp, r, dec_attn_mask=..., mems=..., head_mask=..., output_attentions=...): # -> Any:
        ...
    


class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=..., sample_softmax=...) -> None:
        ...
    
    def forward(self, inp): # -> Any | Tensor:
        ...
    


class TransfoXLPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = TransfoXLConfig
    load_tf_weights = ...
    base_model_prefix = ...
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = ..., layer: Optional[int] = ...): # -> Module | Any:
        """
        Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size. Take care of tying
        weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:

            new_num_tokens: (`optional`) int:
                New number of tokens in the embedding matrix. Increasing the size will add newly initialized vectors at
                the end. Reducing the size will remove vectors from the end. If not provided or None: does nothing and
                just returns a pointer to the input tokens ``torch.nn.Embeddings`` Module of the model.
            layer: (`optional`) int:
                Layer of the `AdaptiveEmbedding` where the resizing should be done. Per default the last layer will be
                resized. Be aware that when resizing other than the last layer, you have to ensure that the new
                token(s) in the tokenizer are at the corresponding position.

        Return: ``torch.nn.Embeddings`` Pointer to the input tokens Embeddings Module of the model
        """
        ...
    


@dataclass
class TransfoXLModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see :obj:`mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
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
    last_hidden_state: torch.FloatTensor
    mems: List[torch.FloatTensor] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class TransfoXLSequenceClassifierOutputWithPast(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see :obj:`mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
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
    mems: List[torch.FloatTensor] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class TransfoXLLMHeadModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        losses (:obj:`torch.FloatTensor` of shape `(batch_size, sequence_length-1)`, `optional`, returned when ``labels`` is provided)
            Language modeling losses (not reduced).
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token after SoftMax).
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see :obj:`mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
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
    losses: Optional[torch.FloatTensor] = ...
    prediction_scores: torch.FloatTensor = ...
    mems: List[torch.FloatTensor] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...
    @property
    def logits(self): # -> FloatTensor:
        ...
    


TRANSFO_XL_START_DOCSTRING = ...
TRANSFO_XL_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare Bert Model transformer outputting raw hidden-states without any specific head on top.", TRANSFO_XL_START_DOCSTRING)
class TransfoXLModel(TransfoXLPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> AdaptiveEmbedding:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    def backward_compatible(self): # -> None:
        ...
    
    def reset_memory_length(self, mem_len): # -> None:
        ...
    
    def init_mems(self, bsz): # -> list[Unknown] | None:
        ...
    
    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TransfoXLModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., mems=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


@add_start_docstrings("""
    The Transformer-XL Model with a language modeling head on top (adaptive softmax with weights tied to the adaptive
    input embeddings)
    """, TRANSFO_XL_START_DOCSTRING)
class TransfoXLLMHeadModel(TransfoXLPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def tie_weights(self): # -> None:
        """
        Run this to be sure output and input (adaptive) softmax weights are tied
        """
        ...
    
    def reset_memory_length(self, mem_len): # -> None:
        ...
    
    def init_mems(self, bsz): # -> list[Unknown] | None:
        ...
    
    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TransfoXLLMHeadModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., mems=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | TransfoXLLMHeadModelOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        ...
    
    def get_output_embeddings(self): # -> Tensor | Module:
        """Double-check if you are using adaptive softmax."""
        ...
    
    def prepare_inputs_for_generation(self, input_ids, past=..., **model_kwargs): # -> dict[Unknown, Unknown]:
        ...
    


@add_start_docstrings("""
    The Transformer-XL Model transformer with a sequence classification head on top (linear layer).

    :class:`~transformers.TransfoXLForSequenceClassification` uses the last token in order to do the classification, as
    other causal models (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    :obj:`pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each
    row. If no :obj:`pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot
    guess the padding tokens when :obj:`inputs_embeds` are passed instead of :obj:`input_ids`, it does the same (take
    the last value in each row of the batch).
    """, TRANSFO_XL_START_DOCSTRING)
class TransfoXLForSequenceClassification(TransfoXLPreTrainedModel):
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TransfoXLSequenceClassifierOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., mems=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


