

from ...configuration_utils import PretrainedConfig

logger = ...
CANINE_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class CanineConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.CanineModel`. It is used to
    instantiate an CANINE model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the CANINE `google/canine-s
    <https://huggingface.co/google/canine-s>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the deep Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoders.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoders.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoders, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 16384):
            The maximum sequence length that this model might ever be used with.
        type_vocab_size (:obj:`int`, `optional`, defaults to 16):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.CanineModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, use gradient checkpointing to save memory at the expense of slower backward pass.
        downsampling_rate (:obj:`int`, `optional`, defaults to 4):
            The rate at which to downsample the original character sequence length before applying the deep Transformer
            encoder.
        upsampling_kernel_size (:obj:`int`, `optional`, defaults to 4):
            The kernel size (i.e. the number of characters in each window) of the convolutional projection layer when
            projecting back from :obj:`hidden_size`*2 to :obj:`hidden_size`.
        num_hash_functions (:obj:`int`, `optional`, defaults to 8):
            The number of hash functions to use. Each hash function has its own embedding matrix.
        num_hash_buckets (:obj:`int`, `optional`, defaults to 16384):
            The number of hash buckets to use.
        local_transformer_stride (:obj:`int`, `optional`, defaults to 128):
            The stride of the local attention of the first shallow Transformer encoder. Defaults to 128 for good
            TPU/XLA memory alignment.

    Example::

        >>> from transformers import CanineModel, CanineConfig

        >>> # Initializing a CANINE google/canine-s style configuration
        >>> configuration = CanineConfig()

        >>> # Initializing a model from the google/canine-s style configuration
        >>> model = CanineModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = ...
    def __init__(self, hidden_size=..., num_hidden_layers=..., num_attention_heads=..., intermediate_size=..., hidden_act=..., hidden_dropout_prob=..., attention_probs_dropout_prob=..., max_position_embeddings=..., type_vocab_size=..., initializer_range=..., layer_norm_eps=..., use_cache=..., is_encoder_decoder=..., pad_token_id=..., bos_token_id=..., eos_token_id=..., downsampling_rate=..., upsampling_kernel_size=..., num_hash_functions=..., num_hash_buckets=..., local_transformer_stride=..., **kwargs) -> None:
        ...
    


