

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
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_mobilebert import MobileBertConfig

logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
def load_tf_weights_in_mobilebert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    ...

class NoNorm(nn.Module):
    def __init__(self, feat_size, eps=...) -> None:
        ...
    
    def forward(self, input_tensor):
        ...
    


NORM2FN = ...
class MobileBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, input_ids=..., token_type_ids=..., position_ids=..., inputs_embeds=...): # -> Any:
        ...
    


class MobileBertSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def transpose_for_scores(self, x):
        ...
    
    def forward(self, query_tensor, key_tensor, value_tensor, attention_mask=..., head_mask=..., output_attentions=...): # -> tuple[Tensor, Unknown | Any] | tuple[Tensor]:
        ...
    


class MobileBertSelfOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, residual_tensor): # -> Any:
        ...
    


class MobileBertAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def forward(self, query_tensor, key_tensor, value_tensor, layer_input, attention_mask=..., head_mask=..., output_attentions=...): # -> Any:
        ...
    


class MobileBertIntermediate(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any | Tensor:
        ...
    


class OutputBottleneck(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, residual_tensor): # -> Any:
        ...
    


class MobileBertOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, intermediate_states, residual_tensor_1, residual_tensor_2): # -> Any:
        ...
    


class BottleneckLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class Bottleneck(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Tuple[Any, ...] | tuple[Any, Any, Unknown, Any] | tuple[Unknown, Unknown, Unknown, Any]:
        ...
    


class FFNOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, residual_tensor): # -> Any:
        ...
    


class FFNLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class MobileBertLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., output_attentions=...): # -> Any:
        ...
    


class MobileBertEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


class MobileBertPooler(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Tensor:
        ...
    


class MobileBertPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class MobileBertLMPredictionHead(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states):
        ...
    


class MobileBertOnlyMLMHead(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, sequence_output): # -> Any:
        ...
    


class MobileBertPreTrainingHeads(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, sequence_output, pooled_output): # -> tuple[Any, Any]:
        ...
    


class MobileBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MobileBertConfig
    pretrained_model_archive_map = ...
    load_tf_weights = ...
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...


@dataclass
class MobileBertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.MobileBertForPreTraining`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
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
    prediction_logits: torch.FloatTensor = ...
    seq_relationship_logits: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


MOBILEBERT_START_DOCSTRING = ...
MOBILEBERT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare MobileBert Model transformer outputting raw hidden-states without any specific head on top.", MOBILEBERT_START_DOCSTRING)
class MobileBertModel(MobileBertPreTrainedModel):
    """
    https://arxiv.org/pdf/2004.02984.pdf
    """
    def __init__(self, config, add_pooling_layer=...) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., output_hidden_states=..., output_attentions=..., return_dict=...):
        ...
    


@add_start_docstrings("""
    MobileBert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
    `next sentence prediction (classification)` head.
    """, MOBILEBERT_START_DOCSTRING)
class MobileBertForPreTraining(MobileBertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddigs): # -> None:
        ...
    
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = ...) -> nn.Embedding:
        ...
    
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MobileBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., next_sentence_label=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | MobileBertForPreTrainingOutput:
        r"""
        labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        next_sentence_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Examples::

            >>> from transformers import MobileBertTokenizer, MobileBertForPreTraining
            >>> import torch

            >>> tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
            >>> model = MobileBertForPreTraining.from_pretrained("google/mobilebert-uncased")

            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            >>> outputs = model(input_ids)

            >>> prediction_logits = outptus.prediction_logits
            >>> seq_relationship_logits = outputs.seq_relationship_logits

        """
        ...
    


@add_start_docstrings("""MobileBert Model with a `language modeling` head on top. """, MOBILEBERT_START_DOCSTRING)
class MobileBertForMaskedLM(MobileBertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddigs): # -> None:
        ...
    
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = ...) -> nn.Embedding:
        ...
    
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | MaskedLMOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        ...
    


class MobileBertOnlyNSPHead(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, pooled_output): # -> Any:
        ...
    


@add_start_docstrings("""MobileBert Model with a `next sentence prediction (classification)` head on top. """, MOBILEBERT_START_DOCSTRING)
class MobileBertForNextSentencePrediction(MobileBertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=..., **kwargs): # -> Any | NextSentencePredictorOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see ``input_ids`` docstring) Indices should be in ``[0, 1]``.

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Examples::

            >>> from transformers import MobileBertTokenizer, MobileBertForNextSentencePrediction
            >>> import torch

            >>> tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
            >>> model = MobileBertForNextSentencePrediction.from_pretrained('google/mobilebert-uncased')

            >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
            >>> encoding = tokenizer(prompt, next_sentence, return_tensors='pt')

            >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
            >>> loss = outputs.loss
            >>> logits = outputs.logits
        """
        ...
    


@add_start_docstrings("""
    MobileBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """, MOBILEBERT_START_DOCSTRING)
class MobileBertForSequenceClassification(MobileBertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


@add_start_docstrings("""
    MobileBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, MOBILEBERT_START_DOCSTRING)
class MobileBertForQuestionAnswering(MobileBertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., start_positions=..., end_positions=..., output_attentions=..., output_hidden_states=..., return_dict=...):
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
    MobileBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """, MOBILEBERT_START_DOCSTRING)
class MobileBertForMultipleChoice(MobileBertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        ...
    


@add_start_docstrings("""
    MobileBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """, MOBILEBERT_START_DOCSTRING)
class MobileBertForTokenClassification(MobileBertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | TokenClassifierOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        ...
    


