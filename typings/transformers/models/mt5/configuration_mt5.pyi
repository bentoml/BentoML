

from ...configuration_utils import PretrainedConfig

logger = ...
class MT5Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.MT5Model` or a
    :class:`~transformers.TFMT5Model`. It is used to instantiate a mT5 model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the mT5 `google/mt5-small <https://huggingface.co/google/mt5-small>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Arguments:
        vocab_size (:obj:`int`, `optional`, defaults to 250112):
            Vocabulary size of the T5 model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.T5Model` or :class:`~transformers.TFT5Model`.
        d_model (:obj:`int`, `optional`, defaults to 512):
            Size of the encoder layers and the pooler layer.
        d_kv (:obj:`int`, `optional`, defaults to 64):
            Size of the key, query, value projections per attention head. :obj:`d_kv` has to be equal to :obj:`d_model
            // num_heads`.
        d_ff (:obj:`int`, `optional`, defaults to 1024):
            Size of the intermediate feed forward layer in each :obj:`T5Block`.
        num_layers (:obj:`int`, `optional`, defaults to 8):
            Number of hidden layers in the Transformer encoder.
        num_decoder_layers (:obj:`int`, `optional`):
            Number of hidden layers in the Transformer decoder. Will use the same value as :obj:`num_layers` if not
            set.
        num_heads (:obj:`int`, `optional`, defaults to 6):
            Number of attention heads for each attention layer in the Transformer encoder.
        relative_attention_num_buckets (:obj:`int`, `optional`, defaults to 32):
            The number of buckets to use for each attention layer.
        dropout_rate (:obj:`float`, `optional`, defaults to 0.1):
            The ratio for all dropout layers.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        initializer_factor (:obj:`float`, `optional`, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        feed_forward_proj (:obj:`string`, `optional`, defaults to :obj:`"gated-gelu"`):
            Type of feed forward layer to be used. Should be one of :obj:`"relu"` or :obj:`"gated-gelu"`.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    """
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(self, vocab_size=..., d_model=..., d_kv=..., d_ff=..., num_layers=..., num_decoder_layers=..., num_heads=..., relative_attention_num_buckets=..., dropout_rate=..., layer_norm_epsilon=..., initializer_factor=..., feed_forward_proj=..., is_encoder_decoder=..., use_cache=..., tokenizer_class=..., tie_word_embeddings=..., pad_token_id=..., eos_token_id=..., decoder_start_token_id=..., **kwargs) -> None:
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
    


