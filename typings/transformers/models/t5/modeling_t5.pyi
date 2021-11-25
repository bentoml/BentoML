

import torch
from torch import nn

from ...file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from .configuration_t5 import T5Config

""" PyTorch T5 model. """
logger = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
T5_PRETRAINED_MODEL_ARCHIVE_LIST = ...
def load_tf_weights_in_t5(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    ...

PARALLELIZE_DOCSTRING = ...
DEPARALLELIZE_DOCSTRING = ...
class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None:
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        ...
    
    def forward(self, hidden_states):
        ...
    


class T5DenseReluDense(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class T5DenseGatedGeluDense(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class T5LayerFF(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states):
        ...
    


class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=...) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def compute_bias(self, query_length, key_length): # -> Any:
        """Compute binned relative position bias"""
        ...
    
    def forward(self, hidden_states, mask=..., key_value_states=..., position_bias=..., past_key_value=..., layer_head_mask=..., query_length=..., use_cache=..., output_attentions=...):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        ...
    


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=...) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., position_bias=..., layer_head_mask=..., past_key_value=..., use_cache=..., output_attentions=...): # -> Any:
        ...
    


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, key_value_states, attention_mask=..., position_bias=..., layer_head_mask=..., past_key_value=..., use_cache=..., query_length=..., output_attentions=...): # -> Any:
        ...
    


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=...) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., position_bias=..., encoder_hidden_states=..., encoder_attention_mask=..., encoder_decoder_position_bias=..., layer_head_mask=..., cross_attn_layer_head_mask=..., past_key_value=..., use_cache=..., output_attentions=..., return_dict=...):
        ...
    


class T5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = T5Config
    load_tf_weights = ...
    base_model_prefix = ...
    is_parallelizable = ...
    @property
    def dummy_inputs(self): # -> dict[str, Tensor]:
        ...
    


class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=...) -> None:
        ...
    
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=...): # -> None:
        ...
    
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def deparallelize(self): # -> None:
        ...
    
    def get_input_embeddings(self):
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    def forward(self, input_ids=..., attention_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., inputs_embeds=..., head_mask=..., cross_attn_head_mask=..., past_key_values=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


T5_START_DOCSTRING = ...
T5_INPUTS_DOCSTRING = ...
T5_ENCODER_INPUTS_DOCSTRING = ...
__HEAD_MASK_WARNING_MSG = ...
@add_start_docstrings("The bare T5 Model transformer outputting raw hidden-states" "without any specific head on top.", T5_START_DOCSTRING)
class T5Model(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = ...
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: T5Config) -> None:
        ...
    
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=...): # -> None:
        ...
    
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self): # -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    def get_encoder(self): # -> T5Stack:
        ...
    
    def get_decoder(self): # -> T5Stack:
        ...
    
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., decoder_input_ids=..., decoder_attention_mask=..., head_mask=..., decoder_head_mask=..., cross_attn_head_mask=..., encoder_outputs=..., past_key_values=..., inputs_embeds=..., decoder_inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        Returns:

        Example::

            >>> from transformers import T5Tokenizer, T5Model

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5Model.from_pretrained('t5-small')

            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

            >>> last_hidden_states = outputs.last_hidden_state
        """
        ...
    


@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
class T5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = ...
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=...): # -> None:
        ...
    
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self): # -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def get_encoder(self): # -> T5Stack:
        ...
    
    def get_decoder(self): # -> T5Stack:
        ...
    
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., decoder_input_ids=..., decoder_attention_mask=..., head_mask=..., decoder_head_mask=..., cross_attn_head_mask=..., encoder_outputs=..., past_key_values=..., inputs_embeds=..., decoder_inputs_embeds=..., labels=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        ...
    
    def prepare_inputs_for_generation(self, input_ids, past=..., attention_mask=..., head_mask=..., decoder_head_mask=..., cross_attn_head_mask=..., use_cache=..., encoder_outputs=..., **kwargs): # -> dict[str, Unknown]:
        ...
    
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor): # -> Tensor:
        ...
    


@add_start_docstrings("The bare T5 Model transformer outputting encoder's raw hidden-states without any specific head on top.", T5_START_DOCSTRING)
class T5EncoderModel(T5PreTrainedModel):
    authorized_missing_keys = ...
    def __init__(self, config: T5Config) -> None:
        ...
    
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=...): # -> None:
        ...
    
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self): # -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    def get_encoder(self): # -> T5Stack:
        ...
    
    @add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any:
        r"""
        Returns:

        Example::

            >>> from transformers import T5Tokenizer, T5EncoderModel
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5EncoderModel.from_pretrained('t5-small')
            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        ...
    


