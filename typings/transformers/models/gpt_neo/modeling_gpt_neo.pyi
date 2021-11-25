

from torch import nn

from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from .configuration_gpt_neo import GPTNeoConfig

""" PyTorch GPT Neo model. """
logger = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
GPT_NEO_PRETRAINED_MODEL_ARCHIVE_LIST = ...
_CHECKPOINT_FOR_DOC = ...
def load_tf_weights_in_gpt_neo(model, config, gpt_neo_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    ...

class GPTNeoAttentionMixin:
    """
    A few attention related utilities for attention modules in GPT Neo, to be used as a mixin.
    """
    @staticmethod
    def create_local_attention_mask(batch_size, seq_length, window_size, device, attention_mask=...):
        ...
    


class GPTNeoSelfAttention(nn.Module, GPTNeoAttentionMixin):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., layer_past=..., head_mask=..., use_cache=..., output_attentions=...): # -> tuple[Any, tuple[Tensor | Any, Tensor | Any] | None, Unknown] | tuple[Any, tuple[Tensor | Any, Tensor | Any] | None]:
        ...
    


class GPTNeoLocalSelfAttention(nn.Module, GPTNeoAttentionMixin):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask, layer_past=..., head_mask=..., use_cache=..., output_attentions=...): # -> tuple[Any, Unknown] | tuple[Any]:
        ...
    


class GPTNeoAttention(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...
    
    def forward(self, hidden_states, layer_past=..., attention_mask=..., head_mask=..., use_cache=..., output_attentions=...): # -> Any:
        ...
    


class GPTNeoMLP(nn.Module):
    def __init__(self, intermediate_size, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class GPTNeoBlock(nn.Module):
    def __init__(self, config, layer_id) -> None:
        ...
    
    def forward(self, hidden_states, layer_past=..., attention_mask=..., head_mask=..., use_cache=..., output_attentions=...): # -> Any:
        ...
    


class GPTNeoPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GPTNeoConfig
    load_tf_weights = ...
    base_model_prefix = ...
    def __init__(self, *inputs, **kwargs) -> None:
        ...
    


GPT_NEO_START_DOCSTRING = ...
GPT_NEO_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare GPT Neo Model transformer outputting raw hidden-states without any specific head on top.", GPT_NEO_START_DOCSTRING)
class GPTNeoModel(GPTNeoPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(GPT_NEO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPastAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., past_key_values=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


@add_start_docstrings("""
    The GPT Neo Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """, GPT_NEO_START_DOCSTRING)
class GPTNeoForCausalLM(GPTNeoPreTrainedModel):
    _keys_to_ignore_on_load_missing = ...
    _keys_to_ignore_on_save = ...
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    def prepare_inputs_for_generation(self, input_ids, past=..., **kwargs): # -> dict[str, Unknown | None]:
        ...
    
    @add_start_docstrings_to_model_forward(GPT_NEO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., past_key_values=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | CausalLMOutputWithPast:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        ...
    


@add_start_docstrings("""
    The GPTNeo Model transformer with a sequence classification head on top (linear layer).

    :class:`~transformers.GPTNeoForSequenceClassification` uses the last token in order to do the classification, as
    other causal models (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    :obj:`pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each
    row. If no :obj:`pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot
    guess the padding tokens when :obj:`inputs_embeds` are passed instead of :obj:`input_ids`, it does the same (take
    the last value in each row of the batch).
    """, GPT_NEO_START_DOCSTRING)
class GPTNeoForSequenceClassification(GPTNeoPreTrainedModel):
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(GPT_NEO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., past_key_values=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


