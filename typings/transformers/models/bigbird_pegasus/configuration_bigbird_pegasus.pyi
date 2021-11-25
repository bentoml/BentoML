

from ...configuration_utils import PretrainedConfig

logger = ...
BIGBIRD_PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class BigBirdPegasusConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.BigBirdPegasusModel`. It is
    used to instantiate an BigBirdPegasus model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the BigBirdPegasus
    `google/bigbird-pegasus-large-arxiv <https://huggingface.co/google/bigbird-pegasus-large-arxiv>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 96103):
            Vocabulary size of the BigBirdPegasus model. Defines the number of different tokens that can be represented
            by the :obj:`inputs_ids` passed when calling :class:`~transformers.BigBirdPegasusModel`.
        d_model (:obj:`int`, `optional`, defaults to 1024):
            Dimension of the layers and the pooler layer.
        encoder_layers (:obj:`int`, `optional`, defaults to 16):
            Number of encoder layers.
        decoder_layers (:obj:`int`, `optional`, defaults to 16):
            Number of decoder layers.
        encoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 4096):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 1024 or 2048 or 4096).
        init_std (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        encoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the encoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        decoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the decoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        attention_type (:obj:`str`, `optional`, defaults to :obj:`"block_sparse"`)
            Whether to use block sparse attention (with n complexity) as introduced in paper or original attention
            layer (with n^2 complexity) in encoder. Possible values are :obj:`"original_full"` and
            :obj:`"block_sparse"`.
        use_bias (:obj:`bool`, `optional`, defaults to :obj:`False`)
            Whether to use bias in query, key, value.
        block_size (:obj:`int`, `optional`, defaults to 64)
            Size of each block. Useful only when :obj:`attention_type == "block_sparse"`.
        num_random_blocks (:obj:`int`, `optional`, defaults to 3)
            Each query is going to attend these many number of random blocks. Useful only when :obj:`attention_type ==
            "block_sparse"`.
        scale_embeddings (:obj:`bool`, `optional`, defaults to :obj:`True`)
            Whether to rescale embeddings with (hidden_size ** 0.5).
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.

        Example::

        >>> from transformers import BigBirdPegasusModel, BigBirdPegasusConfig

        >>> # Initializing a BigBirdPegasus bigbird-pegasus-base style configuration
        >>> configuration = BigBirdPegasusConfig()

        >>> # Initializing a model from the bigbird-pegasus-base style configuration
        >>> model = BigBirdPegasusModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(self, vocab_size=..., max_position_embeddings=..., encoder_layers=..., encoder_ffn_dim=..., encoder_attention_heads=..., decoder_layers=..., decoder_ffn_dim=..., decoder_attention_heads=..., encoder_layerdrop=..., decoder_layerdrop=..., use_cache=..., is_encoder_decoder=..., activation_function=..., d_model=..., dropout=..., attention_dropout=..., activation_dropout=..., init_std=..., decoder_start_token_id=..., classifier_dropout=..., scale_embedding=..., gradient_checkpointing=..., pad_token_id=..., bos_token_id=..., eos_token_id=..., attention_type=..., block_size=..., num_random_blocks=..., use_bias=..., **kwargs) -> None:
        ...
    
    @property
    def num_attention_heads(self) -> int:
        ...
    
    @property
    def hidden_size(self) -> int:
        ...
    
    @property
    def attention_probs_dropout_prob(self) -> float:
        ...
    


