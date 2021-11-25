

from typing import Any, Dict, Optional, Tuple, Union

import tensorflow as tf

from ...file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput
from ...modeling_tf_utils import TFPreTrainedModel, keras_serializable
from .configuration_wav2vec2 import Wav2Vec2Config

""" TensorFlow Wav2Vec2 model. """
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
TF_WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST = ...
LARGE_NEGATIVE = ...
def input_values_processing(func, config, input_values, **kwargs):
    """
    Process the input of each TensorFlow model including the booleans. In case of a list of symbolic inputs, each input
    has to be named accordingly to the parameters name, i.e. :obj:`input_values = tf.keras.Input(shape=(128,),
    dtype='float32', name="input_values")` otherwise the order of the tensors will not be guaranteed during the
    training.

    Args:
        func (:obj:`callable`):
            The callable function of the TensorFlow model.
        config (:class:`~transformers.PretrainedConfig`):
            The config of the running model.
        **kwargs:
            The inputs of the model.

    Returns:
        Two lists, one for the missing layers, and another one for the unexpected layers.
    """
    ...

class TFWav2Vec2GroupNorm(tf.keras.layers.Layer):
    """
    From tensorflow-addons https://www.tensorflow.org/addons/api_docs/python/tfa/layers/GroupNormalization
    """
    def __init__(self, groups: int = ..., axis: int = ..., epsilon: float = ..., center: bool = ..., scale: bool = ..., beta_initializer: tf.keras.initializers.Initializer = ..., gamma_initializer: tf.keras.initializers.Initializer = ..., beta_regularizer: tf.keras.regularizers.Regularizer = ..., gamma_regularizer: tf.keras.regularizers.Regularizer = ..., beta_constraint: tf.keras.constraints.Constraint = ..., gamma_constraint: tf.keras.constraints.Constraint = ..., **kwargs) -> None:
        ...
    
    def build(self, input_shape): # -> None:
        ...
    
    def call(self, inputs):
        ...
    
    def get_config(self): # -> dict[str, int | Unknown | float | bool]:
        ...
    
    def compute_output_shape(self, input_shape):
        ...
    


class TFWav2Vec2WeightNormConv1D(tf.keras.layers.Conv1D):
    """Adapted from https://www.tensorflow.org/probability/api_docs/python/tfp/layers/weight_norm/WeightNorm"""
    def __init__(self, filters, kernel_size, groups, explicit_padding, **kwargs) -> None:
        ...
    
    def build(self, input_shape): # -> None:
        ...
    
    def call(self, inputs):
        ...
    


class TFWav2Vec2NoLayerNormConvLayer(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, layer_id: int = ..., **kwargs: Any) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...
    


class TFWav2Vec2LayerNormConvLayer(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, layer_id: int = ..., **kwargs: Any) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...
    


class TFWav2Vec2GroupNormConvLayer(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, layer_id: int = ..., **kwargs: Any) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...
    


class TFWav2Vec2PositionalConvEmbedding(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs: Any) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...
    


class TFWav2Vec2SamePadLayer(tf.keras.layers.Layer):
    def __init__(self, num_conv_pos_embeddings, **kwargs) -> None:
        ...
    
    def call(self, hidden_states):
        ...
    


class TFWav2Vec2FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs: Any) -> None:
        ...
    
    def call(self, input_values):
        ...
    


class TFWav2Vec2FeatureProjection(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...
    


class TFWav2Vec2Attention(tf.keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = ..., is_decoder: bool = ..., bias: bool = ..., **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, key_value_states: Optional[tf.Tensor] = ..., past_key_value: Optional[Tuple[Tuple[tf.Tensor]]] = ..., attention_mask: Optional[tf.Tensor] = ..., layer_head_mask: Optional[tf.Tensor] = ..., training=...) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """Input shape: Batch x Time x Channel"""
        ...
    


class TFWav2Vec2FeedForward(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...
    


class TFWav2Vec2EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: Optional[tf.Tensor] = ..., output_attentions: Optional[bool] = ..., training: bool = ...) -> Tuple[tf.Tensor]:
        ...
    


class TFWav2Vec2EncoderLayerStableLayerNorm(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: Optional[tf.Tensor] = ..., output_attentions: Optional[bool] = ..., training: bool = ...) -> Tuple[tf.Tensor]:
        ...
    


class TFWav2Vec2Encoder(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: Optional[tf.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        ...
    


class TFWav2Vec2EncoderStableLayerNorm(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: Optional[tf.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        ...
    


@keras_serializable
class TFWav2Vec2MainLayer(tf.keras.layers.Layer):
    config_class = Wav2Vec2Config
    def __init__(self, config: Wav2Vec2Config, **kwargs) -> None:
        ...
    
    def build(self, input_shape: tf.TensorShape): # -> None:
        ...
    
    def call(self, input_values: tf.Tensor, attention_mask: Optional[tf.Tensor] = ..., token_type_ids: Optional[tf.Tensor] = ..., position_ids: Optional[tf.Tensor] = ..., head_mask: Optional[tf.Tensor] = ..., inputs_embeds: Optional[tf.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: bool = ..., **kwargs: Any): # -> TFBaseModelOutput:
        ...
    


class TFWav2Vec2PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Wav2Vec2Config
    base_model_prefix = ...
    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        ...
    
    @tf.function
    def serving(self, inputs):
        ...
    


WAV_2_VEC_2_START_DOCSTRING = ...
WAV_2_VEC_2_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare TFWav2Vec2 Model transformer outputing raw hidden-states without any specific head on top.", WAV_2_VEC_2_START_DOCSTRING)
class TFWav2Vec2Model(TFWav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_values: tf.Tensor, attention_mask: Optional[tf.Tensor] = ..., token_type_ids: Optional[tf.Tensor] = ..., position_ids: Optional[tf.Tensor] = ..., head_mask: Optional[tf.Tensor] = ..., inputs_embeds: Optional[tf.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: bool = ...) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        """

        Returns:

        Example::

            >>> from transformers import Wav2Vec2Processor, TFWav2Vec2Model
            >>> from datasets import load_dataset
            >>> import soundfile as sf

            >>> processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            >>> model = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

            >>> def map_to_array(batch):
            >>>     speech, _ = sf.read(batch["file"])
            >>>     batch["speech"] = speech
            >>>     return batch

            >>> ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
            >>> ds = ds.map(map_to_array)

            >>> input_values = processor(ds["speech"][0], return_tensors="tf").input_values  # Batch size 1
            >>> hidden_states = model(input_values).last_hidden_state
        """
        ...
    
    def serving_output(self, output): # -> TFBaseModelOutput:
        ...
    


@add_start_docstrings("""TFWav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC). """, WAV_2_VEC_2_START_DOCSTRING)
class TFWav2Vec2ForCTC(TFWav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config, *inputs, **kwargs) -> None:
        ...
    
    def freeze_feature_extractor(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature extractor so that its parameter
        will not be updated during training.
        """
        ...
    
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFCausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_values: tf.Tensor, attention_mask: Optional[tf.Tensor] = ..., token_type_ids: Optional[tf.Tensor] = ..., position_ids: Optional[tf.Tensor] = ..., head_mask: Optional[tf.Tensor] = ..., inputs_embeds: Optional[tf.Tensor] = ..., output_attentions: Optional[bool] = ..., labels: Optional[tf.Tensor] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[TFCausalLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (:obj:`tf.Tensor` or :obj:`np.ndarray` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_values`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``

        Returns:

        Example::

            >>> import tensorflow as tf
            >>> from transformers import Wav2Vec2Processor, TFWav2Vec2ForCTC
            >>> from datasets import load_dataset
            >>> import soundfile as sf

            >>> processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            >>> model = TFWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

            >>> def map_to_array(batch):
            >>>     speech, _ = sf.read(batch["file"])
            >>>     batch["speech"] = speech
            >>>     return batch

            >>> ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
            >>> ds = ds.map(map_to_array)

            >>> input_values = processor(ds["speech"][0], return_tensors="tf").input_values # Batch size 1
            >>> logits = model(input_values).logits
            >>> predicted_ids = tf.argmax(logits, axis=-1)

            >>> transcription = processor.decode(predicted_ids[0])

            >>> # compute loss
            >>> target_transcription = "A MAN SAID TO THE UNIVERSE SIR I EXIST"

            >>> # wrap processor as target processor to encode labels
            >>> with processor.as_target_processor():
            >>>     labels = processor(transcription, return_tensors="tf").input_ids

            >>> loss = model(input_values, labels=labels).loss
        """
        ...
    
    def serving_output(self, output: TFCausalLMOutput) -> TFCausalLMOutput:
        ...
    


