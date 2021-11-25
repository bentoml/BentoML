

from ...configuration_utils import PretrainedConfig

logger = ...
PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class ProphetNetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.ProphetNetModel`. It is used
    to instantiate a ProphetNet model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        activation_dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for activations inside the fully connected layer.
        activation_function (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the ProphetNET model. Defines the number of different tokens that can be represented by
            the :obj:`inputs_ids` passed when calling :class:`~transformers.ProphetNetModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        num_encoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of encoder layers.
        num_encoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the ``intermediate`` (often named feed-forward) layer in decoder.
        num_decoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of decoder layers.
        num_decoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        add_cross_attention (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether cross-attention layers should be added to the model.
        is_encoder_decoder (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether this is an encoder/decoder model.
        pad_token_id (:obj:`int`, `optional`, defaults to 1)
            Padding token id.
        bos_token_id (:obj:`int`, `optional`, defaults to 0)
            Beginning of stream token id.
        eos_token_id (:obj:`int`, `optional`, defaults to 2)
            End of stream token id.
        ngram (:obj:`int`, `optional`, defaults to 2)
            Number of future tokens to predict. Set to 1 to be same as traditional Language model to predict next first
            token.
        num_buckets (:obj:`int`, `optional`, defaults to 32)
            The number of buckets to use for each attention layer. This is for relative position calculation. See the
            `T5 paper <see https://arxiv.org/abs/1910.10683>`__ for more details.
        relative_max_distance (:obj:`int`, `optional`, defaults to 128)
            Relative distances greater than this number will be put into the last same bucket. This is for relative
            position calculation. See the `T5 paper <see https://arxiv.org/abs/1910.10683>`__ for more details.
        disable_ngram_loss (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether be trained predicting only the next first token.
        eps (:obj:`float`, `optional`, defaults to 0.0):
            Controls the ``epsilon`` parameter value for label smoothing in the loss calculation. If set to 0, no label
            smoothing is performed.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
    """
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(self, activation_dropout=..., activation_function=..., vocab_size=..., hidden_size=..., encoder_ffn_dim=..., num_encoder_layers=..., num_encoder_attention_heads=..., decoder_ffn_dim=..., num_decoder_layers=..., num_decoder_attention_heads=..., attention_dropout=..., dropout=..., max_position_embeddings=..., init_std=..., is_encoder_decoder=..., add_cross_attention=..., decoder_start_token_id=..., ngram=..., num_buckets=..., relative_max_distance=..., disable_ngram_loss=..., gradient_checkpointing=..., eps=..., use_cache=..., pad_token_id=..., bos_token_id=..., eos_token_id=..., **kwargs) -> None:
        ...
    
    @property
    def num_attention_heads(self) -> int:
        ...
    
    @property
    def num_hidden_layers(self) -> int:
        ...
    


