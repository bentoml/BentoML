

from typing import Any, Dict, Iterable, Mapping, Optional

from ... import PreTrainedTokenizer, TensorType
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast

logger = ...
GPT_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class GPTNeoConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.GPTNeoModel`. It is used to
    instantiate a GPT Neo model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the GPTNeo `gpt-neo-1.3B
    <https://huggingface.co/EleutherAI/gpt-neo-1.3B>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50257):
            Vocabulary size of the GPT Neo model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.GPTNeoModel`. Vocabulary size of the model.
            Defines the different tokens that can be represented by the `inputs_ids` passed to the forward method of
            :class:`~transformers.GPTNeoModel`.
        attention_types (:obj:`List`, `optional`, defaults to :obj:`[[["global", "local"], 12]]`):
            The type of attention for each layer in a :obj:`List` of the following format :obj:`[[["attention_type"],
            num_layerss]]` e.g. for a 24 layer model :obj:`[[["global"], 24]]` or :obj:`[[["global", "local"], 12]]`
            Choose the value of ``attention_type`` from :obj:`["global", "local"]`
        hidden_size (:obj:`int`, `optional`, defaults to 2048):
            Dimensionality of the encoder layers and the pooler layer.
        num_layers (:obj:`int`, `optional`, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 8192):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        activation_function (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        embed_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.GPTNeoModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_epsilon (:obj:`float`, `optional`, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.

        Example::

            >>> from transformers import GPTNeoModel, GPTNeoConfig

            >>> # Initializing a GPTNeo EleutherAI/gpt-neo-1.3B style configuration
            >>> configuration = GPTNeoConfig()

            >>> # Initializing a model from the EleutherAI/gpt-neo-1.3B style configuration
            >>> model = GPTNeoModel(configuration)

            >>> # Accessing the model configuration
            >>> configuration = model.config
    """
    model_type = ...
    def __init__(self, vocab_size=..., max_position_embeddings=..., hidden_size=..., num_layers=..., attention_types=..., num_heads=..., intermediate_size=..., window_size=..., activation_function=..., resid_dropout=..., embed_dropout=..., attention_dropout=..., layer_norm_epsilon=..., initializer_range=..., summary_type=..., summary_use_proj=..., summary_activation=..., summary_proj_to_labels=..., summary_first_dropout=..., gradient_checkpointing=..., use_cache=..., bos_token_id=..., eos_token_id=..., **kwargs) -> None:
        ...
    
    @staticmethod
    def expand_attention_types_params(attention_types):
        ...
    
    @property
    def num_attention_heads(self):
        ...
    
    @property
    def num_hidden_layers(self):
        ...
    


def custom_unfold(input, dimension, size, step):
    """Custom torch.Tensor.unfold implementation to enable the export to ONNX."""
    ...

def custom_get_block_length_and_num_blocks(seq_length, window_size):
    """
    Custom implementation for GPTNeoAttentionMixin._get_block_length_and_num_blocks to enable the export to ONNX as
    original implmentation uses Python variables and control flow.
    """
    ...

class GPTNeoOnnxConfig(OnnxConfigWithPast):
    def __init__(self, config: PretrainedConfig, task: str = ..., use_past: bool = ...) -> None:
        ...
    
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        ...
    
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        ...
    
    def generate_dummy_inputs(self, tokenizer: PreTrainedTokenizer, batch_size: int = ..., seq_length: int = ..., is_pair: bool = ..., framework: Optional[TensorType] = ...) -> Mapping[str, Any]:
        ...
    
    @staticmethod
    def flatten_output_collection_property(name: str, field: Iterable[Any]) -> Dict[str, Any]:
        ...
    


