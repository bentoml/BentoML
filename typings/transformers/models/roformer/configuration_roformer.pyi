

from ...configuration_utils import PretrainedConfig

logger = ...
ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class RoFormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.RoFormerModel`. It is used to
    instantiate an RoFormer model according to the specified arguments, defining the model architecture. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the RoFormer
    `junnyu/roformer_chinese_base <https://huggingface.co/junnyu/roformer_chinese_base>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50000):
            Vocabulary size of the RoFormer model. Defines the number of different tokens that can be represented by
            the :obj:`inputs_ids` passed when calling :class:`~transformers.RoFormerModel` or
            :class:`~transformers.TFRoFormerModel`.
        embedding_size (:obj:`int`, `optional`, defaults to None):
            Dimensionality of the encoder layers and the pooler layer. Defaults to the :obj:`hidden_size` if not
            provided.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 1536):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 1536).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.RoFormerModel`
            or :class:`~transformers.TFRoFormerModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.
        rotary_value (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not apply rotary position embeddings on value layer.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, use gradient checkpointing to save memory at the expense of slower backward pass.

    Example::

        >>> from transformers import RoFormerModel, RoFormerConfig

        >>> # Initializing a RoFormer junnyu/roformer_chinese_base style configuration
        >>> configuration = RoFormerConfig()

        >>> # Initializing a model from the junnyu/roformer_chinese_base style configuration
        >>> model = RoFormerModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = ...
    def __init__(self, vocab_size=..., embedding_size=..., hidden_size=..., num_hidden_layers=..., num_attention_heads=..., intermediate_size=..., hidden_act=..., hidden_dropout_prob=..., attention_probs_dropout_prob=..., max_position_embeddings=..., type_vocab_size=..., initializer_range=..., layer_norm_eps=..., pad_token_id=..., gradient_checkpointing=..., rotary_value=..., use_cache=..., **kwargs) -> None:
        ...
    


