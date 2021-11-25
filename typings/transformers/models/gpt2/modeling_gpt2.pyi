

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
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from .configuration_gpt2 import GPT2Config

"""PyTorch OpenAI GPT-2 model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = ...
def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    ...

class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=...) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def forward(self, hidden_states, layer_past=..., attention_mask=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., use_cache=..., output_attentions=...): # -> tuple[Any, tuple[Tensor | Any, Tensor | Any] | None, Unknown | Any] | tuple[Any, tuple[Tensor | Any, Tensor | Any] | None]:
        ...
    


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class GPT2Block(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, layer_past=..., attention_mask=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., use_cache=..., output_attentions=...): # -> Any:
        ...
    


class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GPT2Config
    load_tf_weights = ...
    base_model_prefix = ...
    is_parallelizable = ...
    def __init__(self, *inputs, **kwargs) -> None:
        ...
    


@dataclass
class GPT2DoubleHeadsModelOutput(ModelOutput):
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
        past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of length :obj:`config.n_layers`, containing tuples of tensors of shape :obj:`(batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            GPT2Attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """
    loss: Optional[torch.FloatTensor] = ...
    mc_loss: Optional[torch.FloatTensor] = ...
    logits: torch.FloatTensor = ...
    mc_logits: torch.FloatTensor = ...
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


GPT2_START_DOCSTRING = ...
GPT2_INPUTS_DOCSTRING = ...
PARALLELIZE_DOCSTRING = ...
DEPARALLELIZE_DOCSTRING = ...
@add_start_docstrings("The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.", GPT2_START_DOCSTRING)
class GPT2Model(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ...
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
    
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPastAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., past_key_values=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., encoder_hidden_states=..., encoder_attention_mask=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


@add_start_docstrings("""
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """, GPT2_START_DOCSTRING)
class GPT2LMHeadModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=...): # -> None:
        ...
    
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self): # -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    def prepare_inputs_for_generation(self, input_ids, past=..., **kwargs): # -> dict[str, Unknown | None]:
        ...
    
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., past_key_values=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., encoder_hidden_states=..., encoder_attention_mask=..., labels=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | CausalLMOutputWithCrossAttentions:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        ...
    


@add_start_docstrings("""
The GPT2 Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the
input embeddings, the classification head takes as input the input of a specified classification token index in the
input sequence).
""", GPT2_START_DOCSTRING)
class GPT2DoubleHeadsModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=...): # -> None:
        ...
    
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self): # -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    def prepare_inputs_for_generation(self, input_ids, past=..., **kwargs): # -> dict[str, Unknown | None]:
        ...
    
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=GPT2DoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., past_key_values=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., mc_token_ids=..., labels=..., mc_labels=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., **kwargs): # -> Any | GPT2DoubleHeadsModelOutput:
        r"""
        mc_token_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_choices)`, `optional`, default to index of the last token of the input):
            Index of the classification token in each input sequence. Selected in the range ``[0, input_ids.size(-1) -
            1[``.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size - 1]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size - 1]``
        mc_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices]`` where `num_choices` is the size of the second dimension of the input tensors. (see
            `input_ids` above)

        Return:

        Example::

            >>> import torch
            >>> from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

            >>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            >>> model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

            >>> # Add a [CLS] to the vocabulary (we should train it also!)
            >>> num_added_tokens = tokenizer.add_special_tokens({'cls_token': '[CLS]'})

            >>> embedding_layer = model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size

            >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
            >>> encoded_choices = [tokenizer.encode(s) for s in choices]
            >>> cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

            >>> input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
            >>> mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

            >>> outputs = model(input_ids, mc_token_ids=mc_token_ids)
            >>> lm_logits = outputs.logits
            >>> mc_logits = outputs.mc_logits

        """
        ...
    


@add_start_docstrings("""
    The GPT2 Model transformer with a sequence classification head on top (linear layer).

    :class:`~transformers.GPT2ForSequenceClassification` uses the last token in order to do the classification, as
    other causal models (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    :obj:`pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each
    row. If no :obj:`pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot
    guess the padding tokens when :obj:`inputs_embeds` are passed instead of :obj:`input_ids`, it does the same (take
    the last value in each row of the batch).
    """, GPT2_START_DOCSTRING)
class GPT2ForSequenceClassification(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="microsoft/DialogRPT-updown", output_type=SequenceClassifierOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., past_key_values=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


