

from ...configuration_utils import PretrainedConfig

logger = ...
IBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class IBertConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.IBertModel`. It is used to
    instantiate a I-BERT model according to the specified arguments,

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the I-BERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.IBertModel`
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
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
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.IBertModel`
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"absolute"`):
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.
        quant_mode (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to quantize the model or not.
        force_dequant (:obj:`str`, `optional`, defaults to :obj:`"none"`):
            Force dequantize specific nonlinear layer. Dequatized layers are then executed with full precision.
            :obj:`"none"`, :obj:`"gelu"`, :obj:`"softmax"`, :obj:`"layernorm"` and :obj:`"nonlinear"` are supported. As
            deafult, it is set as :obj:`"none"`, which does not dequantize any layers. Please specify :obj:`"gelu"`,
            :obj:`"softmax"`, or :obj:`"layernorm"` to dequantize GELU, Softmax, or LayerNorm, respectively.
            :obj:`"nonlinear"` will dequantize all nonlinear layers, i.e., GELU, Softmax, and LayerNorm.
    """
    model_type = ...
    def __init__(self, vocab_size=..., hidden_size=..., num_hidden_layers=..., num_attention_heads=..., intermediate_size=..., hidden_act=..., hidden_dropout_prob=..., attention_probs_dropout_prob=..., max_position_embeddings=..., type_vocab_size=..., initializer_range=..., layer_norm_eps=..., pad_token_id=..., bos_token_id=..., eos_token_id=..., position_embedding_type=..., quant_mode=..., force_dequant=..., **kwargs) -> None:
        ...
    


