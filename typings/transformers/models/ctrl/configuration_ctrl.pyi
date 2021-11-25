

from ...configuration_utils import PretrainedConfig

logger = ...
CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class CTRLConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.CTRLModel` or a
    :class:`~transformers.TFCTRLModel`. It is used to instantiate a CTRL model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the `ctrl <https://huggingface.co/ctrl>`__ architecture from SalesForce.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 246534):
            Vocabulary size of the CTRL model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.CTRLModel` or
            :class:`~transformers.TFCTRLModel`.
        n_positions (:obj:`int`, `optional`, defaults to 256):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_ctx (:obj:`int`, `optional`, defaults to 256):
            Dimensionality of the causal mask (usually same as n_positions).
        n_embd (:obj:`int`, `optional`, defaults to 1280):
            Dimensionality of the embeddings and hidden states.
        dff (:obj:`int`, `optional`, defaults to 8192):
            Dimensionality of the inner dimension of the feed forward networks (FFN).
        n_layer (:obj:`int`, `optional`, defaults to 48):
            Number of hidden layers in the Transformer encoder.
        n_head (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        resid_pdrop (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (:obj:`int`, `optional`, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (:obj:`float`, `optional`, defaults to 1e-6):
            The epsilon to use in the layer normalization layers
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models).


    Examples::

        >>> from transformers import CTRLModel, CTRLConfig

        >>> # Initializing a CTRL configuration
        >>> configuration = CTRLConfig()

        >>> # Initializing a model from the configuration
        >>> model = CTRLModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(self, vocab_size=..., n_positions=..., n_ctx=..., n_embd=..., dff=..., n_layer=..., n_head=..., resid_pdrop=..., embd_pdrop=..., attn_pdrop=..., layer_norm_epsilon=..., initializer_range=..., summary_type=..., summary_use_proj=..., summary_activation=..., summary_proj_to_labels=..., summary_first_dropout=..., use_cache=..., **kwargs) -> None:
        ...
    
    @property
    def max_position_embeddings(self):
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
    


