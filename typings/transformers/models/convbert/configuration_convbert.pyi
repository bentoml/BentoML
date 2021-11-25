

from ...configuration_utils import PretrainedConfig

logger = ...
CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class ConvBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.ConvBertModel`. It is used to
    instantiate an ConvBERT model according to the specified arguments, defining the model architecture. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the ConvBERT `conv-bert-base
    <https://huggingface.co/YituTech/conv-bert-base>`__ architecture. Configuration objects inherit from
    :class:`~transformers.PretrainedConfig` and can be used to control the model outputs. Read the documentation from
    :class:`~transformers.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the ConvBERT model. Defines the number of different tokens that can be represented by
            the :obj:`inputs_ids` passed when calling :class:`~transformers.ConvBertModel` or
            :class:`~transformers.TFConvBertModel`.
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
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.ConvBertModel`
            or :class:`~transformers.TFConvBertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        head_ratio (:obj:`int`, `optional`, defaults to 2):
            Ratio gamma to reduce the number of attention heads.
        num_groups (:obj:`int`, `optional`, defaults to 1):
            The number of groups for grouped linear layers for ConvBert model
        conv_kernel_size (:obj:`int`, `optional`, defaults to 9):
            The size of the convolutional kernel.


    Example::
        >>> from transformers import ConvBertModel, ConvBertConfig
        >>> # Initializing a ConvBERT convbert-base-uncased style configuration
        >>> configuration = ConvBertConfig()
        >>> # Initializing a model from the convbert-base-uncased style configuration
        >>> model = ConvBertModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = ...
    def __init__(self, vocab_size=..., hidden_size=..., is_encoder_decoder=..., num_hidden_layers=..., num_attention_heads=..., intermediate_size=..., hidden_act=..., hidden_dropout_prob=..., attention_probs_dropout_prob=..., max_position_embeddings=..., type_vocab_size=..., initializer_range=..., layer_norm_eps=..., pad_token_id=..., bos_token_id=..., eos_token_id=..., embedding_size=..., head_ratio=..., conv_kernel_size=..., num_groups=..., **kwargs) -> None:
        ...
    


