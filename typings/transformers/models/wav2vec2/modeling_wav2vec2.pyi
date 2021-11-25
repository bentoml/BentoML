

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
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from .configuration_wav2vec2 import Wav2Vec2Config

""" PyTorch Wav2Vec2 model. """
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST = ...
@dataclass
class Wav2Vec2BaseModelOutput(ModelOutput):
    """
    Output type of :class:`~transformers.Wav2Vec2BaseModelOutput`, with potential hidden states and attentions.

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        extract_features (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, conv_dim[-1])`):
            Sequence of extracted feature vectors of the last convolutional layer of the model.
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
    last_hidden_state: torch.FloatTensor = ...
    extract_features: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class Wav2Vec2ForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.Wav2Vec2ForPreTrainingOutput`, with potential hidden states and attentions.

    Args:
        loss (`optional`, returned when model is in train mode, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the `official
            paper <https://arxiv.org/pdf/2006.11477.pdf>`__ . (classification) loss.
        projected_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.proj_codevector_dim)`):
            Hidden-states of the model projected to `config.proj_codevector_dim` that can be used to predict the masked
            projected quantized states.
        projected_quantized_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.proj_codevector_dim)`):
            Quantized extracted feature vectors projected to `config.proj_codevector_dim` representing the positive
            target vectors for contrastive loss.
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
    projected_states: torch.FloatTensor = ...
    projected_quantized_states: torch.FloatTensor = ...
    codevector_perplexity: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


class Wav2Vec2NoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any | Tensor:
        ...
    


class Wav2Vec2LayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any | Tensor:
        ...
    


class Wav2Vec2GroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any | Tensor:
        ...
    


class Wav2Vec2PositionalConvEmbedding(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any | Tensor:
        ...
    


class Wav2Vec2SamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings) -> None:
        ...
    
    def forward(self, hidden_states):
        ...
    


class Wav2Vec2FeatureExtractor(nn.Module):
    """Construct the featurs from raw audio waveform"""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, input_values): # -> Any:
        ...
    


class Wav2Vec2FeatureProjection(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> tuple[Any, Any]:
        ...
    


class Wav2Vec2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = ..., is_decoder: bool = ..., bias: bool = ...) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor] = ..., past_key_value: Optional[Tuple[torch.Tensor]] = ..., attention_mask: Optional[torch.Tensor] = ..., layer_head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        ...
    


class Wav2Vec2FeedForward(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class Wav2Vec2EncoderLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., output_attentions=...): # -> tuple[Any, Any] | tuple[Any]:
        ...
    


class Wav2Vec2EncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., output_attentions=...): # -> tuple[Unknown, Any] | tuple[Unknown]:
        ...
    


class Wav2Vec2Encoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


class Wav2Vec2EncoderStableLayerNorm(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


class Wav2Vec2GumbelVectorQuantizer(nn.Module):
    """
    Vector quantization using gumbel softmax. See `CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX
    <https://arxiv.org/pdf/1611.01144.pdf>`__ for more information.
    """
    def __init__(self, config) -> None:
        ...
    
    def set_temperature(self, temperature: int): # -> None:
        ...
    
    def forward(self, hidden_states, mask_time_indices=...): # -> tuple[Unknown | Any, Tensor]:
        ...
    


class Wav2Vec2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Wav2Vec2Config
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...


WAV_2_VEC_2_START_DOCSTRING = ...
WAV_2_VEC_2_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare Wav2Vec2 Model transformer outputting raw hidden-states without any specific head on top.", WAV_2_VEC_2_START_DOCSTRING)
class Wav2Vec2Model(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_values, attention_mask=..., mask_time_indices=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | Wav2Vec2BaseModelOutput:
        """

        Returns:

        Example::

            >>> from transformers import Wav2Vec2Processor, Wav2Vec2Model
            >>> from datasets import load_dataset
            >>> import soundfile as sf

            >>> processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            >>> model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

            >>> def map_to_array(batch):
            >>>     speech, _ = sf.read(batch["file"])
            >>>     batch["speech"] = speech
            >>>     return batch

            >>> ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
            >>> ds = ds.map(map_to_array)

            >>> input_values = processor(ds["speech"][0], return_tensors="pt").input_values  # Batch size 1
            >>> hidden_states = model(input_values).last_hidden_state
        """
        ...
    


@add_start_docstrings("""Wav2Vec2 Model with a quantizer and `VQ` head on top. """, WAV_2_VEC_2_START_DOCSTRING)
class Wav2Vec2ForPreTraining(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config) -> None:
        ...
    
    def set_gumbel_temperature(self, temperature: int): # -> None:
        """
        Set the Gumbel softmax temperature to a given value. Only necessary for training
        """
        ...
    
    def freeze_feature_extractor(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature extractor so that its parameters
        will not be updated during training.
        """
        ...
    
    @staticmethod
    def compute_contrastive_logits(target_features: torch.FloatTensor, negative_features: torch.FloatTensor, predicted_features: torch.FloatTensor, temperature: int = ...): # -> Tensor:
        """
        Compute logits for contrastive loss based using cosine similarity as the distance measure between
        `[positive_feature, negative_features]` and `[predicted_features]`. Additionally, temperature can be applied.
        """
        ...
    
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_values, attention_mask=..., mask_time_indices=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | Wav2Vec2ForPreTrainingOutput:
        r"""
        mask_time_indices (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
            masked extracted features in `config.proj_codevector_dim` space.

        Returns:

        Example::

            >>> import torch
            >>> from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForPreTraining
            >>> from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
            >>> from datasets import load_dataset
            >>> import soundfile as sf

            >>> feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("patrickvonplaten/wav2vec2-base")
            >>> model = Wav2Vec2ForPreTraining.from_pretrained("patrickvonplaten/wav2vec2-base")


            >>> def map_to_array(batch):
            ...     speech, _ = sf.read(batch["file"])
            ...     batch["speech"] = speech
            ...     return batch


            >>> ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
            >>> ds = ds.map(map_to_array)

            >>> input_values = feature_extractor(ds["speech"][0], return_tensors="pt").input_values  # Batch size 1

            >>> # compute masked indices
            >>> batch_size, raw_sequence_length = input_values.shape
            >>> sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)
            >>> mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.2, mask_length=2, device=model.device)

            >>> with torch.no_grad():
            ...     outputs = model(input_values, mask_time_indices=mask_time_indices)

            >>> # compute cosine similarity between predicted (=projected_states) and target (=projected_quantized_states)
            >>> cosine_sim = torch.cosine_similarity(
            ...     outputs.projected_states, outputs.projected_quantized_states, dim=-1
            ... )

            >>> # show that cosine similarity is much higher than random
            >>> assert cosine_sim[mask_time_indices].mean() > 0.5

            >>> # for contrastive loss training model should be put into train mode
            >>> model.train()
            >>> loss = model(input_values, mask_time_indices=mask_time_indices).loss
        """
        ...
    


@add_start_docstrings("""Wav2Vec2 Model with a `language modeling` head on top. """, WAV_2_VEC_2_START_DOCSTRING)
class Wav2Vec2ForMaskedLM(Wav2Vec2PreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_values, attention_mask=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=...): # -> Any | MaskedLMOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            TODO(PVP): Fill out when adding training

        Returns:

        Example::

            >>> from transformers import Wav2Vec2Processor, Wav2Vec2Model
            >>> from datasets import load_dataset
            >>> import soundfile as sf

            >>> processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            >>> model = Wav2Vec2ForMaskedLM.from_pretrained("facebook/wav2vec2-base-960h")

            >>> def map_to_array(batch):
            >>>     speech, _ = sf.read(batch["file"])
            >>>     batch["speech"] = speech
            >>>     return batch

            >>> ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
            >>> ds = ds.map(map_to_array)

            >>> input_values = processor(ds["speech"][0], return_tensors="pt").input_values  # Batch size 1
            >>> logits = model(input_values).logits

            >>> predicted_ids = torch.argmax(logits, dim=-1)
            >>> transcription = processor.decode(predicted_ids[0])
        """
        ...
    


@add_start_docstrings("""Wav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC). """, WAV_2_VEC_2_START_DOCSTRING)
class Wav2Vec2ForCTC(Wav2Vec2PreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def freeze_feature_extractor(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature extractor so that its parameter
        will not be updated during training.
        """
        ...
    
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_values, attention_mask=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=...): # -> Any | CausalLMOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_length)`, `optional`):
            Labels for connectionist temporal classification. Note that ``target_length`` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in ``[-100, 0, ..., config.vocab_size -
            1]``. All labels set to ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ...,
            config.vocab_size - 1]``.

        Returns:

        Example::

            >>> import torch
            >>> from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
            >>> from datasets import load_dataset
            >>> import soundfile as sf

            >>> processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            >>> model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

            >>> def map_to_array(batch):
            >>>     speech, _ = sf.read(batch["file"])
            >>>     batch["speech"] = speech
            >>>     return batch

            >>> ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
            >>> ds = ds.map(map_to_array)

            >>> input_values = processor(ds["speech"][0], return_tensors="pt").input_values  # Batch size 1
            >>> logits = model(input_values).logits
            >>> predicted_ids = torch.argmax(logits, dim=-1)

            >>> transcription = processor.decode(predicted_ids[0])

            >>> # compute loss
            >>> target_transcription = "A MAN SAID TO THE UNIVERSE SIR I EXIST"

            >>> # wrap processor as target processor to encode labels
            >>> with processor.as_target_processor():
            >>>     labels = processor(target_transcription, return_tensors="pt").input_ids

            >>> loss = model(input_values, labels=labels).loss
        """
        ...
    


