

from typing import Optional, Tuple

import torch
from torch import nn

from ...file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from .configuration_m2m_100 import M2M100Config

""" PyTorch M2M100 model. """
logger = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
M2M_100_PRETRAINED_MODEL_ARCHIVE_LIST = ...
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int): # -> Tensor:
    """
    Shift input ids one token to the right.
    """
    ...

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=...):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    """
    ...

class M2M100SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = ...) -> None:
        ...
    
    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = ...): # -> None:
        ...
    
    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = ...): # -> Tensor:
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        ...
    
    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor = ..., inputs_embeds: torch.Tensor = ..., past_key_values_length: int = ...): # -> Tensor:
        ...
    
    def create_position_ids_from_inputs_embeds(self, inputs_embeds): # -> Tensor:
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        ...
    


class M2M100Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = ..., is_decoder: bool = ..., bias: bool = ...) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor] = ..., past_key_value: Optional[Tuple[torch.Tensor]] = ..., attention_mask: Optional[torch.Tensor] = ..., layer_head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        ...
    


class M2M100EncoderLayer(nn.Module):
    def __init__(self, config: M2M100Config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, layer_head_mask: torch.Tensor, output_attentions: bool = ...): # -> tuple[Tensor, Any] | tuple[Tensor]:
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        ...
    


class M2M100DecoderLayer(nn.Module):
    def __init__(self, config: M2M100Config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., encoder_hidden_states: Optional[torch.Tensor] = ..., encoder_attention_mask: Optional[torch.Tensor] = ..., layer_head_mask: Optional[torch.Tensor] = ..., cross_attn_layer_head_mask: Optional[torch.Tensor] = ..., past_key_value: Optional[Tuple[torch.Tensor]] = ..., output_attentions: Optional[bool] = ..., use_cache: Optional[bool] = ...): # -> tuple[Tensor, Any, Any | None, Any] | tuple[Tensor, Any] | tuple[Tensor, Any, Any | None] | tuple[Tensor]:
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (:obj:`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        ...
    


class M2M100PreTrainedModel(PreTrainedModel):
    config_class = M2M100Config
    base_model_prefix = ...


M2M_100_START_DOCSTRING = ...
M2M_100_GENERATION_EXAMPLE = ...
M2M_100_INPUTS_DOCSTRING = ...
class M2M100Encoder(M2M100PreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`M2M100EncoderLayer`.

    Args:
        config: M2M100Config
        embed_tokens (nn.Embedding): output embedding
    """
    def __init__(self, config: M2M100Config, embed_tokens: Optional[nn.Embedding] = ...) -> None:
        ...
    
    def forward(self, input_ids=..., attention_mask=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.M2M100Tokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
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
    


class M2M100Decoder(M2M100PreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`M2M100DecoderLayer`

    Args:
        config: M2M100Config
        embed_tokens (nn.Embedding): output embedding
    """
    def __init__(self, config: M2M100Config, embed_tokens: Optional[nn.Embedding] = ...) -> None:
        ...
    
    def forward(self, input_ids=..., attention_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., head_mask=..., cross_attn_head_mask=..., past_key_values=..., inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.M2M100Tokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
                Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2
                tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional
                tensors of shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
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
    


@add_start_docstrings("The bare M2M100 Model outputting raw hidden-states without any specific head on top.", M2M_100_START_DOCSTRING)
class M2M100Model(M2M100PreTrainedModel):
    def __init__(self, config: M2M100Config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    def get_encoder(self): # -> M2M100Encoder:
        ...
    
    def get_decoder(self): # -> M2M100Decoder:
        ...
    
    @add_start_docstrings_to_model_forward(M2M_100_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., decoder_input_ids=..., decoder_attention_mask=..., head_mask=..., decoder_head_mask=..., cross_attn_head_mask=..., encoder_outputs=..., past_key_values=..., inputs_embeds=..., decoder_inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


@add_start_docstrings("The M2M100 Model with a language modeling head. Can be used for summarization.", M2M_100_START_DOCSTRING)
class M2M100ForConditionalGeneration(M2M100PreTrainedModel):
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...
    _keys_to_ignore_on_save = ...
    def __init__(self, config: M2M100Config) -> None:
        ...
    
    def get_encoder(self): # -> M2M100Encoder:
        ...
    
    def get_decoder(self): # -> M2M100Decoder:
        ...
    
    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(M2M_100_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(M2M_100_GENERATION_EXAMPLE)
    def forward(self, input_ids=..., attention_mask=..., decoder_input_ids=..., decoder_attention_mask=..., head_mask=..., decoder_head_mask=..., cross_attn_head_mask=..., encoder_outputs=..., past_key_values=..., inputs_embeds=..., decoder_inputs_embeds=..., labels=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | Seq2SeqLMOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Example::

            >>> from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration

            >>> model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
            >>> tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M')

            >>> text_to_translate = "Life is like a box of chocolates"
            >>> model_inputs = tokenizer(text_to_translate, return_tensors='pt')

            >>> # translate to French
            >>> gen_tokens = model.generate( **model_inputs, forced_bos_token_id=tokenizer.get_lang_id("fr"))
            >>> print(tokenizer.batch_decode(gen_tokens, skip_special_tokens=True))
        """
        ...
    
    def prepare_inputs_for_generation(self, decoder_input_ids, past=..., attention_mask=..., head_mask=..., decoder_head_mask=..., cross_attn_head_mask=..., use_cache=..., encoder_outputs=..., **kwargs): # -> dict[str, Unknown | None]:
        ...
    


