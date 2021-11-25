

from ...configuration_utils import PretrainedConfig

logger = ...
CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class CLIPTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.CLIPModel`. It is used to
    instantiate an CLIP model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the CLIP
    `openai/clip-vit-base-patch32 <https://huggingface.co/openai/clip-vit-base-patch32>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 49408):
            Vocabulary size of the CLIP text model. Defines the number of different tokens that can be represented by
            the :obj:`inputs_ids` passed when calling :class:`~transformers.CLIPModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (:obj:`int`, `optional`, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 77):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` :obj:`"quick_gelu"` are supported.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (:obj:`float`, `optional`, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.

    Example::

        >>> from transformers import CLIPTextModel, CLIPTextConfig

        >>> # Initializing a CLIPTextModel with openai/clip-vit-base-patch32 style configuration
        >>> configuration = CLIPTextConfig()

        >>> # Initializing a CLIPTextConfig from the openai/clip-vit-base-patch32 style configuration
        >>> model = CLIPTextModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = ...
    def __init__(self, vocab_size=..., hidden_size=..., intermediate_size=..., num_hidden_layers=..., num_attention_heads=..., max_position_embeddings=..., hidden_act=..., layer_norm_eps=..., dropout=..., attention_dropout=..., initializer_range=..., initializer_factor=..., pad_token_id=..., bos_token_id=..., eos_token_id=..., gradient_checkpointing=..., **kwargs) -> None:
        ...
    


class CLIPVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.CLIPModel`. It is used to
    instantiate an CLIP model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the CLIP
    `openai/clip-vit-base-patch32 <https://huggingface.co/openai/clip-vit-base-patch32>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_size (:obj:`int`, `optional`, defaults to 224):
            The size (resolution) of each image.
        patch_size (:obj:`int`, `optional`, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` :obj:`"quick_gelu"` are supported.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (:obj:`float`, `optional`, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.

    Example::

        >>> from transformers import CLIPVisionModel, CLIPVisionConfig

        >>> # Initializing a CLIPVisionModel with openai/clip-vit-base-patch32 style configuration
        >>> configuration = CLIPVisionConfig()

        >>> # Initializing a CLIPVisionModel model from the openai/clip-vit-base-patch32 style configuration
        >>> model = CLIPVisionModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = ...
    def __init__(self, hidden_size=..., intermediate_size=..., num_hidden_layers=..., num_attention_heads=..., image_size=..., patch_size=..., hidden_act=..., layer_norm_eps=..., dropout=..., attention_dropout=..., initializer_range=..., initializer_factor=..., gradient_checkpointing=..., **kwargs) -> None:
        ...
    


class CLIPConfig(PretrainedConfig):
    r"""
    :class:`~transformers.CLIPConfig` is the configuration class to store the configuration of a
    :class:`~transformers.CLIPModel`. It is used to instantiate CLIP model according to the specified arguments,
    defining the text model and vision model configs.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        text_config_dict (:obj:`dict`, `optional`):
            Dictionary of configuration options used to initialize :class:`~transformers.CLIPTextConfig`.
        vision_config_dict (:obj:`dict`, `optional`):
            Dictionary of configuration options used to initialize :class:`~transformers.CLIPVisionConfig`.
        projection_dim (:obj:`int`, `optional`, defaults to 512):
            Dimentionality of text and vision projection layers.
        kwargs (`optional`):
            Dictionary of keyword arguments.
    """
    model_type = ...
    is_composition = ...
    def __init__(self, text_config_dict=..., vision_config_dict=..., projection_dim=..., **kwargs) -> None:
        ...
    
    @classmethod
    def from_text_vision_configs(cls, text_config: CLIPTextConfig, vision_config: CLIPVisionConfig, **kwargs):
        r"""
        Instantiate a :class:`~transformers.CLIPConfig` (or a derived class) from clip text model configuration and
        clip vision model configuration.

        Returns:
            :class:`CLIPConfig`: An instance of a configuration object
        """
        ...
    
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default
        :meth:`~transformers.PretrainedConfig.to_dict`.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        ...
    


