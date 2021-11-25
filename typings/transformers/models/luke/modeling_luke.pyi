

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from .configuration_luke import LukeConfig

"""PyTorch LUKE model. """
logger = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
LUKE_PRETRAINED_MODEL_ARCHIVE_LIST = ...
@dataclass
class BaseLukeModelOutputWithPooling(BaseModelOutputWithPooling):
    """
    Base class for outputs of the LUKE model.

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        entity_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, entity_length, hidden_size)`):
            Sequence of entity hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) further processed by a
            Linear layer and a Tanh activation function.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of
            each layer plus the initial embedding outputs.
        entity_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, entity_length, hidden_size)`. Entity hidden-states of the model at the output
            of each layer plus the initial entity embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length + entity_length, sequence_length + entity_length)`. Attentions weights after the attention
            softmax, used to compute the weighted average in the self-attention heads.
    """
    entity_last_hidden_state: torch.FloatTensor = ...
    entity_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class BaseLukeModelOutput(BaseModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        entity_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, entity_length, hidden_size)`):
            Sequence of entity hidden-states at the output of the last layer of the model.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        entity_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, entity_length, hidden_size)`. Entity hidden-states of the model at the output
            of each layer plus the initial entity embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    entity_last_hidden_state: torch.FloatTensor = ...
    entity_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class EntityClassificationOutput(ModelOutput):
    """
    Outputs of entity classification models.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of
            each layer plus the initial embedding outputs.
        entity_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, entity_length, hidden_size)`. Entity hidden-states of the model at the output
            of each layer plus the initial entity embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """
    loss: Optional[torch.FloatTensor] = ...
    logits: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    entity_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class EntityPairClassificationOutput(ModelOutput):
    """
    Outputs of entity pair classification models.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of
            each layer plus the initial embedding outputs.
        entity_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, entity_length, hidden_size)`. Entity hidden-states of the model at the output
            of each layer plus the initial entity embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """
    loss: Optional[torch.FloatTensor] = ...
    logits: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    entity_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class EntitySpanClassificationOutput(ModelOutput):
    """
    Outputs of entity span classification models.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of
            each layer plus the initial embedding outputs.
        entity_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, entity_length, hidden_size)`. Entity hidden-states of the model at the output
            of each layer plus the initial entity embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """
    loss: Optional[torch.FloatTensor] = ...
    logits: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    entity_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


class LukeEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, input_ids=..., token_type_ids=..., position_ids=..., inputs_embeds=...): # -> Any:
        ...
    
    def create_position_ids_from_inputs_embeds(self, inputs_embeds): # -> Tensor:
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        ...
    


class LukeEntityEmbeddings(nn.Module):
    def __init__(self, config: LukeConfig) -> None:
        ...
    
    def forward(self, entity_ids: torch.LongTensor, position_ids: torch.LongTensor, token_type_ids: torch.LongTensor = ...): # -> Any:
        ...
    


class LukeSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def transpose_for_scores(self, x):
        ...
    
    def forward(self, word_hidden_states, entity_hidden_states, attention_mask=..., head_mask=..., output_attentions=...): # -> tuple[Tensor, Tensor | None, Unknown | Any] | tuple[Tensor, Tensor | None]:
        ...
    


class LukeSelfOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor): # -> Any:
        ...
    


class LukeAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def prune_heads(self, heads):
        ...
    
    def forward(self, word_hidden_states, entity_hidden_states, attention_mask=..., head_mask=..., output_attentions=...): # -> Any:
        ...
    


class LukeIntermediate(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any | Tensor:
        ...
    


class LukeOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor): # -> Any:
        ...
    


class LukeLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, word_hidden_states, entity_hidden_states, attention_mask=..., head_mask=..., output_attentions=...): # -> Any:
        ...
    
    def feed_forward_chunk(self, attention_output): # -> Any:
        ...
    


class LukeEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, word_hidden_states, entity_hidden_states, attention_mask=..., head_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


class LukePooler(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class LukePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = LukeConfig
    base_model_prefix = ...


LUKE_START_DOCSTRING = ...
LUKE_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare LUKE model transformer outputting raw hidden-states for both word tokens and entities without any specific head on top.", LUKE_START_DOCSTRING)
class LukeModel(LukePreTrainedModel):
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, add_pooling_layer=...) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    def get_entity_embeddings(self): # -> Embedding:
        ...
    
    def set_entity_embeddings(self, value): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(LUKE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseLukeModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., entity_ids=..., entity_attention_mask=..., entity_token_type_ids=..., entity_position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""

        Returns:

        Examples::

            >>> from transformers import LukeTokenizer, LukeModel

            >>> tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
            >>> model = LukeModel.from_pretrained("studio-ousia/luke-base")

            # Compute the contextualized entity representation corresponding to the entity mention "Beyoncé"
            >>> text = "Beyoncé lives in Los Angeles."
            >>> entity_spans = [(0, 7)]  # character-based entity span corresponding to "Beyoncé"

            >>> encoding = tokenizer(text, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")
            >>> outputs = model(**encoding)
            >>> word_last_hidden_state = outputs.last_hidden_state
            >>> entity_last_hidden_state = outputs.entity_last_hidden_state

            # Input Wikipedia entities to obtain enriched contextualized representations of word tokens
            >>> text = "Beyoncé lives in Los Angeles."
            >>> entities = ["Beyoncé", "Los Angeles"]  # Wikipedia entity titles corresponding to the entity mentions "Beyoncé" and "Los Angeles"
            >>> entity_spans = [(0, 7), (17, 28)]  # character-based entity spans corresponding to "Beyoncé" and "Los Angeles"

            >>> encoding = tokenizer(text, entities=entities, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")
            >>> outputs = model(**encoding)
            >>> word_last_hidden_state = outputs.last_hidden_state
            >>> entity_last_hidden_state = outputs.entity_last_hidden_state
        """
        ...
    
    def get_extended_attention_mask(self, word_attention_mask: torch.LongTensor, entity_attention_mask: Optional[torch.LongTensor]):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            word_attention_mask (:obj:`torch.LongTensor`):
                Attention mask for word tokens with ones indicating tokens to attend to, zeros for tokens to ignore.
            entity_attention_mask (:obj:`torch.LongTensor`, `optional`):
                Attention mask for entity tokens with ones indicating tokens to attend to, zeros for tokens to ignore.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        ...
    


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    ...

@add_start_docstrings("""
    The LUKE model with a classification head on top (a linear layer on top of the hidden state of the first entity
    token) for entity classification tasks, such as Open Entity.
    """, LUKE_START_DOCSTRING)
class LukeForEntityClassification(LukePreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(LUKE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=EntityClassificationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., entity_ids=..., entity_attention_mask=..., entity_token_type_ids=..., entity_position_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> tuple[Unknown, Any, Any, Any, Any] | tuple[Any, Any, Any, Any] | EntityClassificationOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)` or :obj:`(batch_size, num_labels)`, `optional`):
            Labels for computing the classification loss. If the shape is :obj:`(batch_size,)`, the cross entropy loss
            is used for the single-label classification. In this case, labels should contain the indices that should be
            in :obj:`[0, ..., config.num_labels - 1]`. If the shape is :obj:`(batch_size, num_labels)`, the binary
            cross entropy loss is used for the multi-label classification. In this case, labels should only contain
            ``[0, 1]``, where 0 and 1 indicate false and true, respectively.

        Returns:

        Examples::

            >>> from transformers import LukeTokenizer, LukeForEntityClassification

            >>> tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")
            >>> model = LukeForEntityClassification.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")

            >>> text = "Beyoncé lives in Los Angeles."
            >>> entity_spans = [(0, 7)]  # character-based entity span corresponding to "Beyoncé"
            >>> inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> logits = outputs.logits
            >>> predicted_class_idx = logits.argmax(-1).item()
            >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
            Predicted class: person
        """
        ...
    


@add_start_docstrings("""
    The LUKE model with a classification head on top (a linear layer on top of the hidden states of the two entity
    tokens) for entity pair classification tasks, such as TACRED.
    """, LUKE_START_DOCSTRING)
class LukeForEntityPairClassification(LukePreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(LUKE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=EntityPairClassificationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., entity_ids=..., entity_attention_mask=..., entity_token_type_ids=..., entity_position_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> tuple[Unknown, Any, Any, Any, Any] | tuple[Any, Any, Any, Any] | EntityPairClassificationOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)` or :obj:`(batch_size, num_labels)`, `optional`):
            Labels for computing the classification loss. If the shape is :obj:`(batch_size,)`, the cross entropy loss
            is used for the single-label classification. In this case, labels should contain the indices that should be
            in :obj:`[0, ..., config.num_labels - 1]`. If the shape is :obj:`(batch_size, num_labels)`, the binary
            cross entropy loss is used for the multi-label classification. In this case, labels should only contain
            ``[0, 1]``, where 0 and 1 indicate false and true, respectively.

        Returns:

        Examples::

            >>> from transformers import LukeTokenizer, LukeForEntityPairClassification

            >>> tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
            >>> model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred")

            >>> text = "Beyoncé lives in Los Angeles."
            >>> entity_spans = [(0, 7), (17, 28)]  # character-based entity spans corresponding to "Beyoncé" and "Los Angeles"
            >>> inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> logits = outputs.logits
            >>> predicted_class_idx = logits.argmax(-1).item()
            >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
            Predicted class: per:cities_of_residence
        """
        ...
    


@add_start_docstrings("""
    The LUKE model with a span classification head on top (a linear layer on top of the hidden states output) for tasks
    such as named entity recognition.
    """, LUKE_START_DOCSTRING)
class LukeForEntitySpanClassification(LukePreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(LUKE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=EntitySpanClassificationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., entity_ids=..., entity_attention_mask=..., entity_token_type_ids=..., entity_position_ids=..., entity_start_positions=..., entity_end_positions=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> tuple[Unknown, Any, Any, Any, Any] | tuple[Any, Any, Any, Any] | EntitySpanClassificationOutput:
        r"""
        entity_start_positions (:obj:`torch.LongTensor`):
            The start positions of entities in the word token sequence.

        entity_end_positions (:obj:`torch.LongTensor`):
            The end positions of entities in the word token sequence.

        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, entity_length)` or :obj:`(batch_size, entity_length, num_labels)`, `optional`):
            Labels for computing the classification loss. If the shape is :obj:`(batch_size, entity_length)`, the cross
            entropy loss is used for the single-label classification. In this case, labels should contain the indices
            that should be in :obj:`[0, ..., config.num_labels - 1]`. If the shape is :obj:`(batch_size, entity_length,
            num_labels)`, the binary cross entropy loss is used for the multi-label classification. In this case,
            labels should only contain ``[0, 1]``, where 0 and 1 indicate false and true, respectively.

        Returns:

        Examples::

            >>> from transformers import LukeTokenizer, LukeForEntitySpanClassification

            >>> tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")
            >>> model = LukeForEntitySpanClassification.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")

            >>> text = "Beyoncé lives in Los Angeles"

            # List all possible entity spans in the text
            >>> word_start_positions = [0, 8, 14, 17, 21]  # character-based start positions of word tokens
            >>> word_end_positions = [7, 13, 16, 20, 28]  # character-based end positions of word tokens
            >>> entity_spans = []
            >>> for i, start_pos in enumerate(word_start_positions):
            ...     for end_pos in word_end_positions[i:]:
            ...         entity_spans.append((start_pos, end_pos))

            >>> inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> logits = outputs.logits
            >>> predicted_class_indices = logits.argmax(-1).squeeze().tolist()
            >>> for span, predicted_class_idx in zip(entity_spans, predicted_class_indices):
            ...     if predicted_class_idx != 0:
            ...        print(text[span[0]:span[1]], model.config.id2label[predicted_class_idx])
            Beyoncé PER
            Los Angeles LOC
        """
        ...
    


