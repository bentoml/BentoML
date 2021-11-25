

from typing import Optional, Tuple

import torch
from torch import nn

from ...file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import BaseModelOutput, CausalLMOutput
from ...modeling_utils import PreTrainedModel
from .configuration_hubert import HubertConfig

""" PyTorch Hubert model. """
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class HubertNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any | Tensor:
        ...
    


class HubertLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any | Tensor:
        ...
    


class HubertGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any | Tensor:
        ...
    


class HubertPositionalConvEmbedding(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any | Tensor:
        ...
    


class HubertSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings) -> None:
        ...
    
    def forward(self, hidden_states):
        ...
    


class HubertFeatureExtractor(nn.Module):
    """Construct the featurs from raw audio waveform"""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, input_values): # -> Any:
        ...
    


class HubertFeatureProjection(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class HubertAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = ..., is_decoder: bool = ..., bias: bool = ...) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor] = ..., past_key_value: Optional[Tuple[torch.Tensor]] = ..., attention_mask: Optional[torch.Tensor] = ..., layer_head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        ...
    


class HubertFeedForward(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class HubertEncoderLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., output_attentions=...): # -> tuple[Any, Any] | tuple[Any]:
        ...
    


class HubertEncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., output_attentions=...): # -> tuple[Unknown, Any] | tuple[Unknown]:
        ...
    


class HubertEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


class HubertEncoderStableLayerNorm(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


class HubertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = HubertConfig
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...


HUBERT_START_DOCSTRING = ...
HUBERT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare Hubert Model transformer outputting raw hidden-states without any specific head on top.", HUBERT_START_DOCSTRING)
class HubertModel(HubertPreTrainedModel):
    def __init__(self, config: HubertConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(HUBERT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_values, attention_mask=..., mask_time_indices=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | BaseModelOutput:
        """

        Returns:

        Example::

            >>> from transformers import Wav2Vec2Processor, HubertModel
            >>> from datasets import load_dataset
            >>> import soundfile as sf

            >>> processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
            >>> model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

            >>> def map_to_array(batch):
            ...     speech, _ = sf.read(batch["file"])
            ...     batch["speech"] = speech
            ...     return batch

            >>> ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
            >>> ds = ds.map(map_to_array)

            >>> input_values = processor(ds["speech"][0], return_tensors="pt").input_values  # Batch size 1
            >>> hidden_states = model(input_values).last_hidden_state
        """
        ...
    


@add_start_docstrings("""Hubert Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC). """, HUBERT_START_DOCSTRING)
class HubertForCTC(HubertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def freeze_feature_extractor(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature extractor so that its parameter
        will not be updated during training.
        """
        ...
    
    @add_start_docstrings_to_model_forward(HUBERT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutput, config_class=_CONFIG_FOR_DOC)
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
            >>> from transformers import Wav2Vec2Processor, HubertForCTC
            >>> from datasets import load_dataset
            >>> import soundfile as sf

            >>> processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
            >>> model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

            >>> def map_to_array(batch):
            ...     speech, _ = sf.read(batch["file"])
            ...     batch["speech"] = speech
            ...     return batch

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
            ...     labels = processor(target_transcription, return_tensors="pt").input_ids

            >>> loss = model(input_values, labels=labels).loss
        """
        ...
    


