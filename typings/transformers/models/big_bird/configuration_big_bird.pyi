

from ...configuration_utils import PretrainedConfig

logger = ...
BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class BigBirdConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.BigBirdModel`. It is used to
    instantiate an BigBird model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BigBird
    `google/bigbird-roberta-base <https://huggingface.co/google/bigbird-roberta-base>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50358):
            Vocabulary size of the BigBird model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BigBirdModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 4096):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 1024 or 2048 or 4096).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.BigBirdModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.
        attention_type (:obj:`str`, `optional`, defaults to :obj:`"block_sparse"`)
            Whether to use block sparse attention (with n complexity) as introduced in paper or original attention
            layer (with n^2 complexity). Possible values are :obj:`"original_full"` and :obj:`"block_sparse"`.
        use_bias (:obj:`bool`, `optional`, defaults to :obj:`True`)
            Whether to use bias in query, key, value.
        rescale_embeddings (:obj:`bool`, `optional`, defaults to :obj:`False`)
            Whether to rescale embeddings with (hidden_size ** 0.5).
        block_size (:obj:`int`, `optional`, defaults to 64)
            Size of each block. Useful only when :obj:`attention_type == "block_sparse"`.
        num_random_blocks (:obj:`int`, `optional`, defaults to 3)
            Each query is going to attend these many number of random blocks. Useful only when :obj:`attention_type ==
            "block_sparse"`.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.

        Example::

        >>> from transformers import BigBirdModel, BigBirdConfig

        >>> # Initializing a BigBird google/bigbird-roberta-base style configuration
        >>> configuration = BigBirdConfig()

        >>> # Initializing a model from the google/bigbird-roberta-base style configuration
        >>> model = BigBirdModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = ...
    def __init__(self, vocab_size=..., hidden_size=..., num_hidden_layers=..., num_attention_heads=..., intermediate_size=..., hidden_act=..., hidden_dropout_prob=..., attention_probs_dropout_prob=..., max_position_embeddings=..., type_vocab_size=..., initializer_range=..., layer_norm_eps=..., use_cache=..., is_encoder_decoder=..., pad_token_id=..., bos_token_id=..., eos_token_id=..., sep_token_id=..., attention_type=..., use_bias=..., rescale_embeddings=..., block_size=..., num_random_blocks=..., gradient_checkpointing=..., **kwargs) -> None:
        ...
    


