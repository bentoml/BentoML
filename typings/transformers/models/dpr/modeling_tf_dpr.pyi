

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import tensorflow as tf

from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import TFBaseModelOutputWithPooling
from ...modeling_tf_utils import TFPreTrainedModel
from .configuration_dpr import DPRConfig

""" TensorFlow DPR model for Open Domain Question Answering."""
logger = ...
_CONFIG_FOR_DOC = ...
TF_DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = ...
TF_DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = ...
TF_DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST = ...
@dataclass
class TFDPRContextEncoderOutput(ModelOutput):
    r"""
    Class for outputs of :class:`~transformers.TFDPRContextEncoder`.

    Args:
        pooler_output: (:obj:``tf.Tensor`` of shape ``(batch_size, embeddings_size)``):
            The DPR encoder outputs the `pooler_output` that corresponds to the context representation. Last layer
            hidden-state of the first token of the sequence (classification token) further processed by a Linear layer.
            This output is to be used to embed contexts for nearest neighbors queries with questions embeddings.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    pooler_output: tf.Tensor = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...


@dataclass
class TFDPRQuestionEncoderOutput(ModelOutput):
    """
    Class for outputs of :class:`~transformers.TFDPRQuestionEncoder`.

    Args:
        pooler_output: (:obj:``tf.Tensor`` of shape ``(batch_size, embeddings_size)``):
            The DPR encoder outputs the `pooler_output` that corresponds to the question representation. Last layer
            hidden-state of the first token of the sequence (classification token) further processed by a Linear layer.
            This output is to be used to embed questions for nearest neighbors queries with context embeddings.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    pooler_output: tf.Tensor = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...


@dataclass
class TFDPRReaderOutput(ModelOutput):
    """
    Class for outputs of :class:`~transformers.TFDPRReaderEncoder`.

    Args:
        start_logits: (:obj:``tf.Tensor`` of shape ``(n_passages, sequence_length)``):
            Logits of the start index of the span for each passage.
        end_logits: (:obj:``tf.Tensor`` of shape ``(n_passages, sequence_length)``):
            Logits of the end index of the span for each passage.
        relevance_logits: (:obj:`tf.Tensor`` of shape ``(n_passages, )``):
            Outputs of the QA classifier of the DPRReader that corresponds to the scores of each passage to answer the
            question, compared to all the other passages.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    start_logits: tf.Tensor = ...
    end_logits: tf.Tensor = ...
    relevance_logits: tf.Tensor = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...


class TFDPREncoderLayer(tf.keras.layers.Layer):
    base_model_prefix = ...
    def __init__(self, config: DPRConfig, **kwargs) -> None:
        ...
    
    def call(self, input_ids: tf.Tensor = ..., attention_mask: Optional[tf.Tensor] = ..., token_type_ids: Optional[tf.Tensor] = ..., inputs_embeds: Optional[tf.Tensor] = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., training: bool = ..., **kwargs) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor, ...]]:
        ...
    
    @property
    def embeddings_size(self) -> int:
        ...
    


class TFDPRSpanPredictorLayer(tf.keras.layers.Layer):
    base_model_prefix = ...
    def __init__(self, config: DPRConfig, **kwargs) -> None:
        ...
    
    def call(self, input_ids: tf.Tensor = ..., attention_mask: Optional[tf.Tensor] = ..., inputs_embeds: Optional[tf.Tensor] = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., training: bool = ..., **kwargs) -> Union[TFDPRReaderOutput, Tuple[tf.Tensor, ...]]:
        ...
    


class TFDPRSpanPredictor(TFPreTrainedModel):
    base_model_prefix = ...
    def __init__(self, config: DPRConfig, **kwargs) -> None:
        ...
    
    def call(self, input_ids: tf.Tensor = ..., attention_mask: Optional[tf.Tensor] = ..., token_type_ids: Optional[tf.Tensor] = ..., inputs_embeds: Optional[tf.Tensor] = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., training: bool = ..., **kwargs) -> Union[TFDPRReaderOutput, Tuple[tf.Tensor, ...]]:
        ...
    


class TFDPREncoder(TFPreTrainedModel):
    base_model_prefix = ...
    def __init__(self, config: DPRConfig, **kwargs) -> None:
        ...
    
    def call(self, input_ids: tf.Tensor = ..., attention_mask: Optional[tf.Tensor] = ..., token_type_ids: Optional[tf.Tensor] = ..., inputs_embeds: Optional[tf.Tensor] = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., training: bool = ..., **kwargs) -> Union[TFDPRReaderOutput, Tuple[tf.Tensor, ...]]:
        ...
    


class TFDPRPretrainedContextEncoder(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DPRConfig
    base_model_prefix = ...


class TFDPRPretrainedQuestionEncoder(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DPRConfig
    base_model_prefix = ...


class TFDPRPretrainedReader(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DPRConfig
    base_model_prefix = ...
    @tf.function(input_signature=[{ "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),"attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask") }])
    def serving(self, inputs):
        ...
    


TF_DPR_START_DOCSTRING = ...
TF_DPR_ENCODERS_INPUTS_DOCSTRING = ...
TF_DPR_READER_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare DPRContextEncoder transformer outputting pooler outputs as context representations.", TF_DPR_START_DOCSTRING)
class TFDPRContextEncoder(TFDPRPretrainedContextEncoder):
    def __init__(self, config: DPRConfig, *args, **kwargs) -> None:
        ...
    
    def get_input_embeddings(self):
        ...
    
    @add_start_docstrings_to_model_forward(TF_DPR_ENCODERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFDPRContextEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask: Optional[tf.Tensor] = ..., token_type_ids: Optional[tf.Tensor] = ..., inputs_embeds: Optional[tf.Tensor] = ..., output_attentions=..., output_hidden_states=..., return_dict=..., training: bool = ..., **kwargs) -> Union[TFDPRContextEncoderOutput, Tuple[tf.Tensor, ...]]:
        r"""
        Return:

        Examples::

            >>> from transformers import TFDPRContextEncoder, DPRContextEncoderTokenizer
            >>> tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
            >>> model = TFDPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', from_pt=True)
            >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors='tf')["input_ids"]
            >>> embeddings = model(input_ids).pooler_output
        """
        ...
    
    def serving_output(self, output): # -> TFDPRContextEncoderOutput:
        ...
    


@add_start_docstrings("The bare DPRQuestionEncoder transformer outputting pooler outputs as question representations.", TF_DPR_START_DOCSTRING)
class TFDPRQuestionEncoder(TFDPRPretrainedQuestionEncoder):
    def __init__(self, config: DPRConfig, *args, **kwargs) -> None:
        ...
    
    def get_input_embeddings(self):
        ...
    
    @add_start_docstrings_to_model_forward(TF_DPR_ENCODERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFDPRQuestionEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask: Optional[tf.Tensor] = ..., token_type_ids: Optional[tf.Tensor] = ..., inputs_embeds: Optional[tf.Tensor] = ..., output_attentions=..., output_hidden_states=..., return_dict=..., training: bool = ..., **kwargs) -> Union[TFDPRQuestionEncoderOutput, Tuple[tf.Tensor, ...]]:
        r"""
        Return:

        Examples::

            >>> from transformers import TFDPRQuestionEncoder, DPRQuestionEncoderTokenizer
            >>> tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
            >>> model = TFDPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base', from_pt=True)
            >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors='tf')["input_ids"]
            >>> embeddings = model(input_ids).pooler_output
        """
        ...
    
    def serving_output(self, output): # -> TFDPRQuestionEncoderOutput:
        ...
    


@add_start_docstrings("The bare DPRReader transformer outputting span predictions.", TF_DPR_START_DOCSTRING)
class TFDPRReader(TFDPRPretrainedReader):
    def __init__(self, config: DPRConfig, *args, **kwargs) -> None:
        ...
    
    def get_input_embeddings(self):
        ...
    
    @add_start_docstrings_to_model_forward(TF_DPR_READER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFDPRReaderOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask: Optional[tf.Tensor] = ..., inputs_embeds: Optional[tf.Tensor] = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict=..., training: bool = ..., **kwargs) -> Union[TFDPRReaderOutput, Tuple[tf.Tensor, ...]]:
        r"""
        Return:

        Examples::

            >>> from transformers import TFDPRReader, DPRReaderTokenizer
            >>> tokenizer = DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')
            >>> model = TFDPRReader.from_pretrained('facebook/dpr-reader-single-nq-base', from_pt=True)
            >>> encoded_inputs = tokenizer(
            ...         questions=["What is love ?"],
            ...         titles=["Haddaway"],
            ...         texts=["'What Is Love' is a song recorded by the artist Haddaway"],
            ...         return_tensors='tf'
            ...     )
            >>> outputs = model(encoded_inputs)
            >>> start_logits = outputs.start_logits
            >>> end_logits = outputs.end_logits
            >>> relevance_logits = outputs.relevance_logits

        """
        ...
    
    def serving_output(self, output): # -> TFDPRReaderOutput:
        ...
    


