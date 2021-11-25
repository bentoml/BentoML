

from dataclasses import dataclass
from typing import List, Optional, Tuple

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
from .configuration_xlnet import XLNetConfig

"""
 PyTorch XLNet model.
"""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
XLNET_PRETRAINED_MODEL_ARCHIVE_LIST = ...
def build_tf_xlnet_to_pytorch_map(model, config, tf_weights=...):
    """
    A map of modules from TF to PyTorch. I use a map to keep the PyTorch model as identical to the original PyTorch
    model as possible.
    """
    ...

def load_tf_weights_in_xlnet(model, config, tf_path):
    """Load tf checkpoints in a pytorch model"""
    ...

class XLNetRelativeAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def prune_heads(self, heads):
        ...
    
    @staticmethod
    def rel_shift(x, klen=...): # -> Tensor:
        """perform relative shift to form the relative attention score."""
        ...
    
    @staticmethod
    def rel_shift_bnij(x, klen=...): # -> Tensor:
        ...
    
    def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, seg_mat=..., attn_mask=..., head_mask=..., output_attentions=...): # -> tuple[Unknown, Unknown]:
        """Core relative positional attention operations."""
        ...
    
    def post_attention(self, h, attn_vec, residual=...): # -> Any:
        """Post-attention processing."""
        ...
    
    def forward(self, h, g, attn_mask_h, attn_mask_g, r, seg_mat, mems=..., target_mapping=..., head_mask=..., output_attentions=...):
        ...
    


class XLNetFeedForward(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, inp): # -> Any:
        ...
    


class XLNetLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, output_h, output_g, attn_mask_h, attn_mask_g, r, seg_mat, mems=..., target_mapping=..., head_mask=..., output_attentions=...): # -> Any:
        ...
    
    def ff_chunk(self, output_x): # -> Any:
        ...
    


class XLNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = XLNetConfig
    load_tf_weights = ...
    base_model_prefix = ...


@dataclass
class XLNetModelOutput(ModelOutput):
    """
    Output type of :class:`~transformers.XLNetModel`.

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_predict, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.

            ``num_predict`` corresponds to ``target_mapping.shape[1]``. If ``target_mapping`` is ``None``, then
            ``num_predict`` corresponds to ``sequence_length``.
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see :obj:`mems` input) to speed up sequential decoding.
            The token ids which have their past given to this model should not be passed as :obj:`input_ids` as they
            have already been computed.
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
    mems: Optional[List[torch.FloatTensor]] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class XLNetLMHeadModelOutput(ModelOutput):
    """
    Output type of :class:`~transformers.XLNetLMHeadModel`.

    Args:
        loss (:obj:`torch.FloatTensor` of shape `(1,)`, `optional`, returned when ``labels`` is provided)
            Language modeling loss (for next-token prediction).
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_predict, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

            ``num_predict`` corresponds to ``target_mapping.shape[1]``. If ``target_mapping`` is ``None``, then
            ``num_predict`` corresponds to ``sequence_length``.
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see :obj:`mems` input) to speed up sequential decoding.
            The token ids which have their past given to this model should not be passed as :obj:`input_ids` as they
            have already been computed.
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
    mems: Optional[List[torch.FloatTensor]] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class XLNetForSequenceClassificationOutput(ModelOutput):
    """
    Output type of :class:`~transformers.XLNetForSequenceClassification`.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see :obj:`mems` input) to speed up sequential decoding.
            The token ids which have their past given to this model should not be passed as :obj:`input_ids` as they
            have already been computed.
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
    mems: Optional[List[torch.FloatTensor]] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class XLNetForTokenClassificationOutput(ModelOutput):
    """
    Output type of :class:`~transformers.XLNetForTokenClassificationOutput`.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see :obj:`mems` input) to speed up sequential decoding.
            The token ids which have their past given to this model should not be passed as :obj:`input_ids` as they
            have already been computed.
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
    mems: Optional[List[torch.FloatTensor]] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class XLNetForMultipleChoiceOutput(ModelOutput):
    """
    Output type of :class:`~transformers.XLNetForMultipleChoice`.

    Args:
        loss (:obj:`torch.FloatTensor` of shape `(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            `num_choices` is the second dimension of the input tensors. (see `input_ids` above).

            Classification scores (before SoftMax).
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see :obj:`mems` input) to speed up sequential decoding.
            The token ids which have their past given to this model should not be passed as :obj:`input_ids` as they
            have already been computed.
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
    mems: Optional[List[torch.FloatTensor]] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class XLNetForQuestionAnsweringSimpleOutput(ModelOutput):
    """
    Output type of :class:`~transformers.XLNetForQuestionAnsweringSimple`.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see :obj:`mems` input) to speed up sequential decoding.
            The token ids which have their past given to this model should not be passed as :obj:`input_ids` as they
            have already been computed.
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
    start_logits: torch.FloatTensor = ...
    end_logits: torch.FloatTensor = ...
    mems: Optional[List[torch.FloatTensor]] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class XLNetForQuestionAnsweringOutput(ModelOutput):
    """
    Output type of :class:`~transformers.XLNetForQuestionAnswering`.

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
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see :obj:`mems` input) to speed up sequential decoding.
            The token ids which have their past given to this model should not be passed as :obj:`input_ids` as they
            have already been computed.
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
    mems: Optional[List[torch.FloatTensor]] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


XLNET_START_DOCSTRING = ...
XLNET_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare XLNet Model transformer outputting raw hidden-states without any specific head on top.", XLNET_START_DOCSTRING)
class XLNetModel(XLNetPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    def create_mask(self, qlen, mlen): # -> Tensor:
        """
        Creates causal attention mask. Float mask where 1.0 indicates masked, 0.0 indicates not-masked.

        Args:
            qlen: Sequence length
            mlen: Mask length

        ::

                  same_length=False:      same_length=True:
                  <mlen > <  qlen >       <mlen > <  qlen >
               ^ [0 0 0 0 0 1 1 1 1]     [0 0 0 0 0 1 1 1 1]
                 [0 0 0 0 0 0 1 1 1]     [1 0 0 0 0 0 1 1 1]
            qlen [0 0 0 0 0 0 0 1 1]     [1 1 0 0 0 0 0 1 1]
                 [0 0 0 0 0 0 0 0 1]     [1 1 1 0 0 0 0 0 1]
               v [0 0 0 0 0 0 0 0 0]     [1 1 1 1 0 0 0 0 0]

        """
        ...
    
    def cache_mem(self, curr_out, prev_mem): # -> Tensor:
        ...
    
    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=...): # -> Tensor:
        ...
    
    def relative_positional_encoding(self, qlen, klen, bsz=...): # -> Tensor:
        ...
    
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=XLNetModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., mems=..., perm_mask=..., target_mapping=..., token_type_ids=..., input_mask=..., head_mask=..., inputs_embeds=..., use_mems=..., output_attentions=..., output_hidden_states=..., return_dict=..., **kwargs):
        ...
    


@add_start_docstrings("""
    XLNet Model with a language modeling head on top (linear layer with weights tied to the input embeddings).
    """, XLNET_START_DOCSTRING)
class XLNetLMHeadModel(XLNetPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    def prepare_inputs_for_generation(self, input_ids, past=..., use_mems=..., **kwargs): # -> dict[str, Tensor | Unknown]:
        ...
    
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=XLNetLMHeadModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., mems=..., perm_mask=..., target_mapping=..., token_type_ids=..., input_mask=..., head_mask=..., inputs_embeds=..., labels=..., use_mems=..., output_attentions=..., output_hidden_states=..., return_dict=..., **kwargs): # -> Any | XLNetLMHeadModelOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_predict)`, `optional`):
            Labels for masked language modeling. :obj:`num_predict` corresponds to :obj:`target_mapping.shape[1]`. If
            :obj:`target_mapping` is :obj`None`, then :obj:`num_predict` corresponds to :obj:`sequence_length`.

            The labels should correspond to the masked input words that should be predicted and depends on
            :obj:`target_mapping`. Note in order to perform standard auto-regressive language modeling a `<mask>` token
            has to be added to the :obj:`input_ids` (see the :obj:`prepare_inputs_for_generation` function and examples
            below)

            Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to ``-100`` are ignored, the
            loss is only computed for labels in ``[0, ..., config.vocab_size]``

        Return:

        Examples::

            >>> from transformers import XLNetTokenizer, XLNetLMHeadModel
            >>> import torch

            >>> tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
            >>> model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased')

            >>> # We show how to setup inputs to predict a next token using a bi-directional context.
            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False)).unsqueeze(0)  # We will predict the masked token
            >>> perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
            >>> perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
            >>> target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
            >>> target_mapping[0, 0, -1] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

            >>> outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
            >>> next_token_logits = outputs[0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]

            >>> # The same way can the XLNetLMHeadModel be used to be trained by standard auto-regressive language modeling.
            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False)).unsqueeze(0)  # We will predict the masked token
            >>> labels = torch.tensor(tokenizer.encode("cute", add_special_tokens=False)).unsqueeze(0)
            >>> assert labels.shape[0] == 1, 'only one word will be predicted'
            >>> perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
            >>> perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token as is done in standard auto-regressive lm training
            >>> target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
            >>> target_mapping[0, 0, -1] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

            >>> outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping, labels=labels)
            >>> loss = outputs.loss
            >>> next_token_logits = outputs.logits  # Logits have shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
        """
        ...
    


@add_start_docstrings("""
    XLNet Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g.
    for GLUE tasks.
    """, XLNET_START_DOCSTRING)
class XLNetForSequenceClassification(XLNetPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=XLNetForSequenceClassificationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., mems=..., perm_mask=..., target_mapping=..., token_type_ids=..., input_mask=..., head_mask=..., inputs_embeds=..., labels=..., use_mems=..., output_attentions=..., output_hidden_states=..., return_dict=..., **kwargs):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in ``[0, ...,
            config.num_labels - 1]``. If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
        """
        ...
    


@add_start_docstrings("""
    XLNet Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, XLNET_START_DOCSTRING)
class XLNetForTokenClassification(XLNetPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=XLNetForTokenClassificationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., mems=..., perm_mask=..., target_mapping=..., token_type_ids=..., input_mask=..., head_mask=..., inputs_embeds=..., labels=..., use_mems=..., output_attentions=..., output_hidden_states=..., return_dict=..., **kwargs): # -> Any | XLNetForTokenClassificationOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices]`` where `num_choices` is the size of the second dimension of the input tensors. (see
            `input_ids` above)
        """
        ...
    


@add_start_docstrings("""
    XLNet Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RACE/SWAG tasks.
    """, XLNET_START_DOCSTRING)
class XLNetForMultipleChoice(XLNetPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=XLNetForMultipleChoiceOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., token_type_ids=..., input_mask=..., attention_mask=..., mems=..., perm_mask=..., target_mapping=..., head_mask=..., inputs_embeds=..., labels=..., use_mems=..., output_attentions=..., output_hidden_states=..., return_dict=..., **kwargs):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        ...
    


@add_start_docstrings("""
    XLNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, XLNET_START_DOCSTRING)
class XLNetForQuestionAnsweringSimple(XLNetPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=XLNetForQuestionAnsweringSimpleOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., mems=..., perm_mask=..., target_mapping=..., token_type_ids=..., input_mask=..., head_mask=..., inputs_embeds=..., start_positions=..., end_positions=..., use_mems=..., output_attentions=..., output_hidden_states=..., return_dict=..., **kwargs):
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
    XLNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, XLNET_START_DOCSTRING)
class XLNetForQuestionAnswering(XLNetPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=XLNetForQuestionAnsweringOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., mems=..., perm_mask=..., target_mapping=..., token_type_ids=..., input_mask=..., head_mask=..., inputs_embeds=..., start_positions=..., end_positions=..., is_impossible=..., cls_index=..., p_mask=..., use_mems=..., output_attentions=..., output_hidden_states=..., return_dict=..., **kwargs):
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

            >>> from transformers import XLNetTokenizer, XLNetForQuestionAnswering
            >>> import torch

            >>> tokenizer =  XLNetTokenizer.from_pretrained('xlnet-base-cased')
            >>> model = XLNetForQuestionAnswering.from_pretrained('xlnet-base-cased')

            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            >>> start_positions = torch.tensor([1])
            >>> end_positions = torch.tensor([3])
            >>> outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)

            >>> loss = outputs.loss
        """
        ...
    


