

from ...configuration_utils import PretrainedConfig

logger = ...
DEIT_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class DeiTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.DeiTModel`. It is used to
    instantiate an DeiT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the DeiT
    `facebook/deit-base-distilled-patch16-224 <https://huggingface.co/facebook/deit-base-distilled-patch16-224>`__
    architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        image_size (:obj:`int`, `optional`, defaults to :obj:`224`):
            The size (resolution) of each image.
        patch_size (:obj:`int`, `optional`, defaults to :obj:`16`):
            The size (resolution) of each patch.
        num_channels (:obj:`int`, `optional`, defaults to :obj:`3`):
            The number of input channels.


    Example::

        >>> from transformers import DeiTModel, DeiTConfig

        >>> # Initializing a DeiT deit-base-distilled-patch16-224 style configuration
        >>> configuration = DeiTConfig()

        >>> # Initializing a model from the deit-base-distilled-patch16-224 style configuration
        >>> model = DeiTModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = ...
    def __init__(self, hidden_size=..., num_hidden_layers=..., num_attention_heads=..., intermediate_size=..., hidden_act=..., hidden_dropout_prob=..., attention_probs_dropout_prob=..., initializer_range=..., layer_norm_eps=..., is_encoder_decoder=..., image_size=..., patch_size=..., num_channels=..., **kwargs) -> None:
        ...
    


