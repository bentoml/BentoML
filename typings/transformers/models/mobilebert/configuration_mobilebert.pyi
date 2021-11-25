

from ...configuration_utils import PretrainedConfig

logger = ...
MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class MobileBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.MobileBertModel` or a
    :class:`~transformers.TFMobileBertModel`. It is used to instantiate a MobileBERT model according to the specified
    arguments, defining the model architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the MobileBERT model. Defines the number of different tokens that can be represented by
            the :obj:`inputs_ids` passed when calling :class:`~transformers.MobileBertModel` or
            :class:`~transformers.TFMobileBertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 4):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 512):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.MobileBertModel`
            or :class:`~transformers.TFMobileBertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.

        pad_token_id (:obj:`int`, `optional`, defaults to 0):
            The ID of the token in the word embedding to use as padding.
        embedding_size (:obj:`int`, `optional`, defaults to 128):
            The dimension of the word embedding vectors.
        trigram_input (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Use a convolution of trigram as input.
        use_bottleneck (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to use bottleneck in BERT.
        intra_bottleneck_size (:obj:`int`, `optional`, defaults to 128):
            Size of bottleneck layer output.
        use_bottleneck_attention (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use attention inputs from the bottleneck transformation.
        key_query_shared_bottleneck (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to use the same linear transformation for query&key in the bottleneck.
        num_feedforward_networks (:obj:`int`, `optional`, defaults to 4):
            Number of FFNs in a block.
        normalization_type (:obj:`str`, `optional`, defaults to :obj:`"no_norm"`):
            The normalization type in MobileBERT.

    Examples::

        >>> from transformers import MobileBertModel, MobileBertConfig

        >>> # Initializing a MobileBERT configuration
        >>> configuration = MobileBertConfig()

        >>> # Initializing a model from the configuration above
        >>> model = MobileBertModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config

    Attributes: pretrained_config_archive_map (Dict[str, str]): A dictionary containing all the available pre-trained
    checkpoints.
    """
    pretrained_config_archive_map = ...
    model_type = ...
    def __init__(self, vocab_size=..., hidden_size=..., num_hidden_layers=..., num_attention_heads=..., intermediate_size=..., hidden_act=..., hidden_dropout_prob=..., attention_probs_dropout_prob=..., max_position_embeddings=..., type_vocab_size=..., initializer_range=..., layer_norm_eps=..., pad_token_id=..., embedding_size=..., trigram_input=..., use_bottleneck=..., intra_bottleneck_size=..., use_bottleneck_attention=..., key_query_shared_bottleneck=..., num_feedforward_networks=..., normalization_type=..., classifier_activation=..., **kwargs) -> None:
        ...
    


