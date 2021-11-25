

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from .configuration_dpr import DPRConfig

""" PyTorch DPR model for Open Domain Question Answering."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = ...
DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = ...
DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST = ...
@dataclass
class DPRContextEncoderOutput(ModelOutput):
    """
    Class for outputs of :class:`~transformers.DPRQuestionEncoder`.

    Args:
        pooler_output: (:obj:``torch.FloatTensor`` of shape ``(batch_size, embeddings_size)``):
            The DPR encoder outputs the `pooler_output` that corresponds to the context representation. Last layer
            hidden-state of the first token of the sequence (classification token) further processed by a Linear layer.
            This output is to be used to embed contexts for nearest neighbors queries with questions embeddings.
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
    pooler_output: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class DPRQuestionEncoderOutput(ModelOutput):
    """
    Class for outputs of :class:`~transformers.DPRQuestionEncoder`.

    Args:
        pooler_output: (:obj:``torch.FloatTensor`` of shape ``(batch_size, embeddings_size)``):
            The DPR encoder outputs the `pooler_output` that corresponds to the question representation. Last layer
            hidden-state of the first token of the sequence (classification token) further processed by a Linear layer.
            This output is to be used to embed questions for nearest neighbors queries with context embeddings.
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
    pooler_output: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class DPRReaderOutput(ModelOutput):
    """
    Class for outputs of :class:`~transformers.DPRQuestionEncoder`.

    Args:
        start_logits: (:obj:``torch.FloatTensor`` of shape ``(n_passages, sequence_length)``):
            Logits of the start index of the span for each passage.
        end_logits: (:obj:``torch.FloatTensor`` of shape ``(n_passages, sequence_length)``):
            Logits of the end index of the span for each passage.
        relevance_logits: (:obj:`torch.FloatTensor`` of shape ``(n_passages, )``):
            Outputs of the QA classifier of the DPRReader that corresponds to the scores of each passage to answer the
            question, compared to all the other passages.
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
    start_logits: torch.FloatTensor
    end_logits: torch.FloatTensor = ...
    relevance_logits: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


class DPREncoder(PreTrainedModel):
    base_model_prefix = ...
    def __init__(self, config: DPRConfig) -> None:
        ...
    
    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = ..., token_type_ids: Optional[Tensor] = ..., inputs_embeds: Optional[Tensor] = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...) -> Union[BaseModelOutputWithPooling, Tuple[Tensor, ...]]:
        ...
    
    @property
    def embeddings_size(self) -> int:
        ...
    
    def init_weights(self): # -> None:
        ...
    


class DPRSpanPredictor(PreTrainedModel):
    base_model_prefix = ...
    def __init__(self, config: DPRConfig) -> None:
        ...
    
    def forward(self, input_ids: Tensor, attention_mask: Tensor, inputs_embeds: Optional[Tensor] = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...) -> Union[DPRReaderOutput, Tuple[Tensor, ...]]:
        ...
    
    def init_weights(self): # -> None:
        ...
    


class DPRPretrainedContextEncoder(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DPRConfig
    load_tf_weights = ...
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...
    def init_weights(self): # -> None:
        ...
    


class DPRPretrainedQuestionEncoder(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DPRConfig
    load_tf_weights = ...
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...
    def init_weights(self): # -> None:
        ...
    


class DPRPretrainedReader(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DPRConfig
    load_tf_weights = ...
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...
    def init_weights(self): # -> None:
        ...
    


DPR_START_DOCSTRING = ...
DPR_ENCODERS_INPUTS_DOCSTRING = ...
DPR_READER_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare DPRContextEncoder transformer outputting pooler outputs as context representations.", DPR_START_DOCSTRING)
class DPRContextEncoder(DPRPretrainedContextEncoder):
    def __init__(self, config: DPRConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(DPR_ENCODERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DPRContextEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[Tensor] = ..., attention_mask: Optional[Tensor] = ..., token_type_ids: Optional[Tensor] = ..., inputs_embeds: Optional[Tensor] = ..., output_attentions=..., output_hidden_states=..., return_dict=...) -> Union[DPRContextEncoderOutput, Tuple[Tensor, ...]]:
        r"""
        Return:

        Examples::

            >>> from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
            >>> tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
            >>> model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
            >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors='pt')["input_ids"]
            >>> embeddings = model(input_ids).pooler_output
        """
        ...
    


@add_start_docstrings("The bare DPRQuestionEncoder transformer outputting pooler outputs as question representations.", DPR_START_DOCSTRING)
class DPRQuestionEncoder(DPRPretrainedQuestionEncoder):
    def __init__(self, config: DPRConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(DPR_ENCODERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DPRQuestionEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[Tensor] = ..., attention_mask: Optional[Tensor] = ..., token_type_ids: Optional[Tensor] = ..., inputs_embeds: Optional[Tensor] = ..., output_attentions=..., output_hidden_states=..., return_dict=...) -> Union[DPRQuestionEncoderOutput, Tuple[Tensor, ...]]:
        r"""
        Return:

        Examples::

            >>> from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
            >>> tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
            >>> model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
            >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors='pt')["input_ids"]
            >>> embeddings = model(input_ids).pooler_output
        """
        ...
    


@add_start_docstrings("The bare DPRReader transformer outputting span predictions.", DPR_START_DOCSTRING)
class DPRReader(DPRPretrainedReader):
    def __init__(self, config: DPRConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(DPR_READER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DPRReaderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[Tensor] = ..., attention_mask: Optional[Tensor] = ..., inputs_embeds: Optional[Tensor] = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict=...) -> Union[DPRReaderOutput, Tuple[Tensor, ...]]:
        r"""
        Return:

        Examples::

            >>> from transformers import DPRReader, DPRReaderTokenizer
            >>> tokenizer = DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')
            >>> model = DPRReader.from_pretrained('facebook/dpr-reader-single-nq-base')
            >>> encoded_inputs = tokenizer(
            ...         questions=["What is love ?"],
            ...         titles=["Haddaway"],
            ...         texts=["'What Is Love' is a song recorded by the artist Haddaway"],
            ...         return_tensors='pt'
            ...     )
            >>> outputs = model(**encoded_inputs)
            >>> start_logits = outputs.stat_logits
            >>> end_logits = outputs.end_logits
            >>> relevance_logits = outputs.relevance_logits

        """
        ...
    


