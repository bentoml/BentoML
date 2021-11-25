

from ...configuration_utils import PretrainedConfig

logger = ...
LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class LukeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.LukeModel`. It is used to
    instantiate a LUKE model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the LUKE model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.LukeModel`.
        entity_vocab_size (:obj:`int`, `optional`, defaults to 500000):
            Entity vocabulary size of the LUKE model. Defines the number of different entities that can be represented
            by the :obj:`entity_ids` passed when calling :class:`~transformers.LukeModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        entity_emb_size (:obj:`int`, `optional`, defaults to 256):
            The number of dimensions of the entity embedding.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.LukeModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        use_entity_aware_attention (:obj:`bool`, defaults to :obj:`True`):
            Whether or not the model should use the entity-aware self-attention mechanism proposed in `LUKE: Deep
            Contextualized Entity Representations with Entity-aware Self-attention (Yamada et al.)
            <https://arxiv.org/abs/2010.01057>`__.

    Examples::

        >>> from transformers import LukeConfig, LukeModel

        >>> # Initializing a LUKE configuration
        >>> configuration = LukeConfig()

        >>> # Initializing a model from the configuration
        >>> model = LukeModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = ...
    def __init__(self, vocab_size=..., entity_vocab_size=..., hidden_size=..., entity_emb_size=..., num_hidden_layers=..., num_attention_heads=..., intermediate_size=..., hidden_act=..., hidden_dropout_prob=..., attention_probs_dropout_prob=..., max_position_embeddings=..., type_vocab_size=..., initializer_range=..., layer_norm_eps=..., gradient_checkpointing=..., use_entity_aware_attention=..., pad_token_id=..., bos_token_id=..., eos_token_id=..., **kwargs) -> None:
        """Constructs LukeConfig."""
        ...
    


