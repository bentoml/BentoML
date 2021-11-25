

from ...configuration_utils import PretrainedConfig

logger = ...
TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class TransfoXLConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.TransfoXLModel` or a
    :class:`~transformers.TFTransfoXLModel`. It is used to instantiate a Transformer-XL model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    similar configuration to that of the `Transformer XL <https://huggingface.co/transfo-xl-wt103>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 267735):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.TransfoXLModel` or
            :class:`~transformers.TFTransfoXLModel`.
        cutoffs (:obj:`List[int]`, `optional`, defaults to :obj:`[20000, 40000, 200000]`):
            Cutoffs for the adaptive softmax.
        d_model (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the model's hidden states.
        d_embed (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the embeddings
        n_head (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        d_head (:obj:`int`, `optional`, defaults to 64):
            Dimensionality of the model's heads.
        d_inner (:obj:`int`, `optional`, defaults to 4096):
            Inner dimension in FF
        div_val (:obj:`int`, `optional`, defaults to 4):
            Divident value for adapative input and softmax
        pre_lnorm (:obj:`boolean`, `optional`, defaults to :obj:`False`):
            Whether or not to apply LayerNorm to the input instead of the output in the blocks.
        n_layer (:obj:`int`, `optional`, defaults to 18):
            Number of hidden layers in the Transformer encoder.
        mem_len (:obj:`int`, `optional`, defaults to 1600):
            Length of the retained previous heads.
        clamp_len (:obj:`int`, `optional`, defaults to 1000):
            Use the same pos embeddings after clamp_len.
        same_length (:obj:`boolean`, `optional`, defaults to :obj:`True`):
            Whether or not to use the same attn length for all tokens
        proj_share_all_but_first (:obj:`boolean`, `optional`, defaults to :obj:`True`):
            True to share all but first projs, False not to share.
        attn_type (:obj:`int`, `optional`, defaults to 0):
            Attention type. 0 for Transformer-XL, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al.
        sample_softmax (:obj:`int`, `optional`, defaults to -1):
            Number of samples in the sampled softmax.
        adaptive (:obj:`boolean`, `optional`, defaults to :obj:`True`):
            Whether or not to use adaptive softmax.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        dropatt (:obj:`float`, `optional`, defaults to 0):
            The dropout ratio for the attention probabilities.
        untie_r (:obj:`boolean`, `optional`, defaults to :obj:`True`):
            Whether ot not to untie relative position biases.
        init (:obj:`str`, `optional`, defaults to :obj:`"normal"`):
            Parameter initializer to use.
        init_range (:obj:`float`, `optional`, defaults to 0.01):
            Parameters initialized by U(-init_range, init_range).
        proj_init_std (:obj:`float`, `optional`, defaults to 0.01):
            Parameters initialized by N(0, init_std)
        init_std (:obj:`float`, `optional`, defaults to 0.02):
            Parameters initialized by N(0, init_std)
        layer_norm_epsilon (:obj:`float`, `optional`, defaults to 1e-5):
            The epsilon to use in the layer normalization layers

    Examples::

        >>> from transformers import TransfoXLConfig, TransfoXLModel

        >>> # Initializing a Transformer XL configuration
        >>> configuration = TransfoXLConfig()

        >>> # Initializing a model from the configuration
        >>> model = TransfoXLModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(self, vocab_size=..., cutoffs=..., d_model=..., d_embed=..., n_head=..., d_head=..., d_inner=..., div_val=..., pre_lnorm=..., n_layer=..., mem_len=..., clamp_len=..., same_length=..., proj_share_all_but_first=..., attn_type=..., sample_softmax=..., adaptive=..., dropout=..., dropatt=..., untie_r=..., init=..., init_range=..., proj_init_std=..., init_std=..., layer_norm_epsilon=..., eos_token_id=..., **kwargs) -> None:
        ...
    
    @property
    def max_position_embeddings(self):
        ...
    
    @property
    def n_token(self):
        ...
    
    @n_token.setter
    def n_token(self, value):
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
    


