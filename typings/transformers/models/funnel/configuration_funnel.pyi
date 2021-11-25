

from ...configuration_utils import PretrainedConfig

logger = ...
FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class FunnelConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.FunnelModel` or a
    :class:`~transformers.TFBertModel`. It is used to instantiate a Funnel Transformer model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the Funnel Transformer `funnel-transformer/small
    <https://huggingface.co/funnel-transformer/small>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the Funnel transformer. Defines the number of different tokens that can be represented
            by the :obj:`inputs_ids` passed when calling :class:`~transformers.FunnelModel` or
            :class:`~transformers.TFFunnelModel`.
        block_sizes (:obj:`List[int]`, `optional`, defaults to :obj:`[4, 4, 4]`):
            The sizes of the blocks used in the model.
        block_repeats (:obj:`List[int]`, `optional`):
            If passed along, each layer of each block is repeated the number of times indicated.
        num_decoder_layers (:obj:`int`, `optional`, defaults to 2):
            The number of layers in the decoder (when not using the base model).
        d_model (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the model's hidden states.
        n_head (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        d_head (:obj:`int`, `optional`, defaults to 64):
            Dimensionality of the model's heads.
        d_inner (:obj:`int`, `optional`, defaults to 3072):
            Inner dimension in the feed-forward blocks.
        hidden_act (:obj:`str` or :obj:`callable`, `optional`, defaults to :obj:`"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for the attention probabilities.
        activation_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout probability used between the two layers of the feed-forward blocks.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 3):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.FunnelModel` or
            :class:`~transformers.TFFunnelModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.1):
            The standard deviation of the `uniform initializer` for initializing all weight matrices in attention
            layers.
        initializer_std (:obj:`float`, `optional`):
            The standard deviation of the `normal initializer` for initializing the embedding matrix and the weight of
            linear layers. Will default to 1 for the embedding matrix and the value given by Xavier initialization for
            linear layers.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-9):
            The epsilon used by the layer normalization layers.
        pooling_type (:obj:`str`, `optional`, defaults to :obj:`"mean"`):
            Possible values are ``"mean"`` or ``"max"``. The way pooling is performed at the beginning of each block.
        attention_type (:obj:`str`, `optional`, defaults to :obj:`"relative_shift"`):
            Possible values are ``"relative_shift"`` or ``"factorized"``. The former is faster on CPU/GPU while the
            latter is faster on TPU.
        separate_cls (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to separate the cls token when applying pooling.
        truncate_seq (:obj:`bool`, `optional`, defaults to :obj:`False`):
            When using ``separate_cls``, whether or not to truncate the last token when pooling, to avoid getting a
            sequence length that is not a multiple of 2.
        pool_q_only (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to apply the pooling only to the query or to query, key and values for the attention layers.
    """
    model_type = ...
    def __init__(self, vocab_size=..., block_sizes=..., block_repeats=..., num_decoder_layers=..., d_model=..., n_head=..., d_head=..., d_inner=..., hidden_act=..., hidden_dropout=..., attention_dropout=..., activation_dropout=..., max_position_embeddings=..., type_vocab_size=..., initializer_range=..., initializer_std=..., layer_norm_eps=..., pooling_type=..., attention_type=..., separate_cls=..., truncate_seq=..., pool_q_only=..., **kwargs) -> None:
        ...
    
    @property
    def hidden_size(self):
        ...
    
    @property
    def num_attention_heads(self):
        ...
    
    @property
    def num_hidden_layers(self):
        ...
    
    @property
    def num_blocks(self):
        ...
    


