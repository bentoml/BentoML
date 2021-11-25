

from torch import nn

from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_ctrl import CTRLConfig

""" PyTorch CTRL model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
CTRL_PRETRAINED_MODEL_ARCHIVE_LIST = ...
def angle_defn(pos, i, d_model_size):
    ...

def positional_encoding(position, d_model_size, dtype): # -> Tensor:
    ...

def scaled_dot_product_attention(q, k, v, mask, attention_mask=..., head_mask=...): # -> tuple[Tensor, Unknown | Tensor]:
    ...

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model_size, num_heads) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def split_into_heads(self, x, batch_size):
        ...
    
    def forward(self, v, k, q, mask, layer_past=..., attention_mask=..., head_mask=..., use_cache=..., output_attentions=...): # -> tuple[Any, Tensor | tuple[None], Unknown | Tensor] | tuple[Any, Tensor | tuple[None]]:
        ...
    


def point_wise_feed_forward_network(d_model_size, dff): # -> Sequential:
    ...

class EncoderLayer(nn.Module):
    def __init__(self, d_model_size, num_heads, dff, rate=...) -> None:
        ...
    
    def forward(self, x, mask, layer_past=..., attention_mask=..., head_mask=..., use_cache=..., output_attentions=...): # -> Any:
        ...
    


class CTRLPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = CTRLConfig
    base_model_prefix = ...


CTRL_START_DOCSTRING = ...
CTRL_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare CTRL Model transformer outputting raw hidden-states without any specific head on top.", CTRL_START_DOCSTRING)
class CTRLModel(CTRLPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., past_key_values=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


@add_start_docstrings("""
    The CTRL Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """, CTRL_START_DOCSTRING)
class CTRLLMHeadModel(CTRLPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    def prepare_inputs_for_generation(self, input_ids, past=..., use_cache=..., **kwargs): # -> dict[str, Unknown]:
        ...
    
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., past_key_values=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | CausalLMOutputWithPast:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        ...
    


@add_start_docstrings("""
    The CTRL Model transformer with a sequence classification head on top (linear layer).
    :class:`~transformers.CTRLForSequenceClassification` uses the last token in order to do the classification, as
    other causal models (e.g. GPT-2) do. Since it does classification on the last token, it requires to know the
    position of the last token. If a :obj:`pad_token_id` is defined in the configuration, it finds the last token that
    is not a padding token in each row. If no :obj:`pad_token_id` is defined, it simply takes the last value in each
    row of the batch. Since it cannot guess the padding tokens when :obj:`inputs_embeds` are passed instead of
    :obj:`input_ids`, it does the same (take the last value in each row of the batch).
    """, CTRL_START_DOCSTRING)
class CTRLForSequenceClassification(CTRLPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., past_key_values=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


