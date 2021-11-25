

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
    CausalLMOutput,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_openai import OpenAIGPTConfig

"""PyTorch OpenAI GPT model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
def load_tf_weights_in_openai_gpt(model, config, openai_checkpoint_folder_path):
    """Load tf pre-trained weights in a pytorch model (from NumPy arrays here)"""
    ...

ACT_FNS = ...
class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=...) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def merge_heads(self, x):
        ...
    
    def split_heads(self, x, k=...):
        ...
    
    def forward(self, x, attention_mask=..., head_mask=..., output_attentions=...): # -> list[Any]:
        ...
    


class MLP(nn.Module):
    def __init__(self, n_state, config) -> None:
        ...
    
    def forward(self, x): # -> Any:
        ...
    


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=...) -> None:
        ...
    
    def forward(self, x, attention_mask=..., head_mask=..., output_attentions=...): # -> Any:
        ...
    


class OpenAIGPTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = OpenAIGPTConfig
    load_tf_weights = ...
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...


@dataclass
class OpenAIGPTDoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided):
            Language modeling loss.
        mc_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`mc_labels` is provided):
            Multiple choice classification loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
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
    mc_loss: Optional[torch.FloatTensor] = ...
    logits: torch.FloatTensor = ...
    mc_logits: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


OPENAI_GPT_START_DOCSTRING = ...
OPENAI_GPT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare OpenAI GPT transformer model outputting raw hidden-states without any specific head on top.", OPENAI_GPT_START_DOCSTRING)
class OpenAIGPTModel(OpenAIGPTPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


@add_start_docstrings("""
    OpenAI GPT Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """, OPENAI_GPT_START_DOCSTRING)
class OpenAIGPTLMHeadModel(OpenAIGPTPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | CausalLMOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        ...
    


@add_start_docstrings("""
OpenAI GPT Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the
input embeddings, the classification head takes as input the input of a specified classification token index in the
input sequence).
""", OPENAI_GPT_START_DOCSTRING)
class OpenAIGPTDoubleHeadsModel(OpenAIGPTPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OpenAIGPTDoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., mc_token_ids=..., labels=..., mc_labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | OpenAIGPTDoubleHeadsModelOutput:
        r"""
        mc_token_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_choices)`, `optional`, default to index of the last token of the input):
            Index of the classification token in each input sequence. Selected in the range ``[0, input_ids.size(-1) -
            1]``.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-1, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        mc_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices]`` where `num_choices` is the size of the second dimension of the input tensors. (see
            `input_ids` above)

        Return:

        Examples::

            >>> from transformers import OpenAIGPTTokenizer, OpenAIGPTDoubleHeadsModel
            >>> import torch

            >>> tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
            >>> model = OpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt')
            >>> tokenizer.add_special_tokens({'cls_token': '[CLS]'})  # Add a [CLS] to the vocabulary (we should train it also!)
            >>> model.resize_token_embeddings(len(tokenizer))

            >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
            >>> input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
            >>> mc_token_ids = torch.tensor([input_ids.size(-1)-1, input_ids.size(-1)-1]).unsqueeze(0)  # Batch size 1

            >>> outputs = model(input_ids, mc_token_ids=mc_token_ids)
            >>> lm_logits = outputs.lm_logits
            >>> mc_logits = outputs.mc_logits
        """
        ...
    


@add_start_docstrings("""
    The Original OpenAI GPT Model transformer with a sequence classification head on top (linear layer).
    :class:`~transformers.OpenAIGPTForSequenceClassification` uses the last token in order to do the classification, as
    other causal models (e.g. GPT-2) do. Since it does classification on the last token, it requires to know the
    position of the last token. If a :obj:`pad_token_id` is defined in the configuration, it finds the last token that
    is not a padding token in each row. If no :obj:`pad_token_id` is defined, it simply takes the last value in each
    row of the batch. Since it cannot guess the padding tokens when :obj:`inputs_embeds` are passed instead of
    :obj:`input_ids`, it does the same (take the last value in each row of the batch).
    """, OPENAI_GPT_START_DOCSTRING)
class OpenAIGPTForSequenceClassification(OpenAIGPTPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


