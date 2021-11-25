

from typing import Optional

import torch
from torch import nn

from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_roformer import RoFormerConfig

""" PyTorch RoFormer model. """
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = ...) -> None:
        ...
    
    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = ...): # -> Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        ...
    


def load_tf_weights_in_roformer(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    ...

class RoFormerEmbeddings(nn.Module):
    """Construct the embeddings from word and token_type embeddings."""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, input_ids=..., token_type_ids=..., inputs_embeds=...): # -> Any:
        ...
    


class RoFormerSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def transpose_for_scores(self, x):
        ...
    
    def forward(self, hidden_states, attention_mask=..., sinusoidal_pos=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., past_key_value=..., output_attentions=...):
        ...
    
    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=...): # -> tuple[Unknown, Unknown, Unknown] | tuple[Unknown, Unknown]:
        ...
    


class RoFormerSelfOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor): # -> Any:
        ...
    


class RoFormerAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., sinusoidal_pos=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., past_key_value=..., output_attentions=...): # -> Any:
        ...
    


class RoFormerIntermediate(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any | Tensor:
        ...
    


class RoFormerOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor): # -> Any:
        ...
    


class RoFormerLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., sinusoidal_pos=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., past_key_value=..., output_attentions=...): # -> Any:
        ...
    
    def feed_forward_chunk(self, attention_output): # -> Any:
        ...
    


class RoFormerEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., past_key_values=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


class RoFormerPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class RoFormerLMPredictionHead(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class RoFormerOnlyMLMHead(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, sequence_output): # -> Any:
        ...
    


class RoFormerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = RoFormerConfig
    load_tf_weights = ...
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...
    _keys_to_ignore_on_load_unexpected = ...


ROFORMER_START_DOCSTRING = ...
ROFORMER_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare RoFormer Model transformer outputting raw hidden-states without any specific head on top.", ROFORMER_START_DOCSTRING)
class RoFormerModel(RoFormerPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPastAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., head_mask=..., inputs_embeds=..., encoder_hidden_states=..., encoder_attention_mask=..., past_key_values=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        ...
    


@add_start_docstrings("""RoFormer Model with a `language modeling` head on top. """, ROFORMER_START_DOCSTRING)
class RoFormerForMaskedLM(RoFormerPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., head_mask=..., inputs_embeds=..., encoder_hidden_states=..., encoder_attention_mask=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | MaskedLMOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.
        """
        ...
    
    def prepare_inputs_for_generation(self, input_ids, attention_mask=..., **model_kwargs): # -> dict[str, Tensor]:
        ...
    


@add_start_docstrings("""RoFormer Model with a `language modeling` head on top for CLM fine-tuning. """, ROFORMER_START_DOCSTRING)
class RoFormerForCausalLM(RoFormerPreTrainedModel):
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., inputs_embeds=..., encoder_hidden_states=..., encoder_attention_mask=..., head_mask=..., cross_attn_head_mask=..., past_key_values=..., labels=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | CausalLMOutputWithCrossAttentions:
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
            ignored (masked), the loss is only computed for the tokens with labels n ``[0, ..., config.vocab_size]``.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).

        Returns:

        Example::

            >>> from transformers import RoFormerTokenizer, RoFormerForCausalLM, RoFormerConfig
            >>> import torch

            >>> tokenizer = RoFormerTokenizer.from_pretrained('junnyu/roformer_chinese_base')
            >>> config = RoFormerConfig.from_pretrained("junnyu/roformer_chinese_base")
            >>> config.is_decoder = True
            >>> model = RoFormerForCausalLM.from_pretrained('junnyu/roformer_chinese_base', config=config)

            >>> inputs = tokenizer("今天天气非常好。", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> prediction_logits = outputs.logits
        """
        ...
    
    def prepare_inputs_for_generation(self, input_ids, past=..., attention_mask=..., **model_kwargs): # -> dict[str, Unknown]:
        ...
    


class RoFormerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, features, **kwargs): # -> Any:
        ...
    


@add_start_docstrings("""
    RoFormer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """, ROFORMER_START_DOCSTRING)
class RoFormerForSequenceClassification(RoFormerPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | SequenceClassifierOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


@add_start_docstrings("""
    RoFormer Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, ROFORMER_START_DOCSTRING)
class RoFormerForMultipleChoice(RoFormerPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        ...
    


@add_start_docstrings("""
    RoFormer Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, ROFORMER_START_DOCSTRING)
class RoFormerForTokenClassification(RoFormerPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | TokenClassifierOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        ...
    


@add_start_docstrings("""
    RoFormer Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, ROFORMER_START_DOCSTRING)
class RoFormerForQuestionAnswering(RoFormerPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., head_mask=..., inputs_embeds=..., start_positions=..., end_positions=..., output_attentions=..., output_hidden_states=..., return_dict=...):
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
    


