

from ...configuration_utils import PretrainedConfig

logger = ...
SPEECH_TO_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class Speech2TextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.Speech2TextModel`. It is used
    to instantiate an Speech2Text model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Speech2Text
    `facebook/s2t-small-librispeech-asr <https://huggingface.co/facebook/s2t-small-librispeech-asr>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50265):
            Vocabulary size of the Speech2Text model. Defines the number of different tokens that can be represented by
            the :obj:`inputs_ids` passed when calling :class:`~transformers.Speech2TextModel`
        d_model (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of encoder layers.
        decoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of decoder layers.
        encoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
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
        max_source_positions (:obj:`int`, `optional`, defaults to 6000):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
        max_target_positions: (:obj:`int`, `optional`, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        num_conv_layers (:obj:`int`, `optional`, defaults to 2):
            Number of 1D convolutional layers in the conv module.
        conv_kernel_sizes (:obj:`Tuple[int]`, `optional`, defaults to :obj:`(5, 5)`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the conv module. The length
            of :obj:`conv_kernel_sizes` has to match :obj:`num_conv_layers`.
        conv_channels (:obj:`int`, `optional`, defaults to 1024):
            An integer defining the number of output channels of each convolution layers except the final one in the
            conv module.
        input_feat_per_channel (:obj:`int`, `optional`, defaults to 80):
            An integer specifying the size of feature vector. This is also the dimensions of log-mel filter-bank
            features.
        input_channels (:obj:`int`, `optional`, defaults to 1):
            An integer specifying number of input channels of the input feature vector.

        Example::

        >>> from transformers import Speech2TextModel, Speech2TextConfig

        >>> # Initializing a Speech2Text s2t_transformer_s style configuration
        >>> configuration = Speech2TextConfig()

        >>> # Initializing a model from the s2t_transformer_s style configuration
        >>> model = Speech2TextModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(self, vocab_size=..., encoder_layers=..., encoder_ffn_dim=..., encoder_attention_heads=..., decoder_layers=..., decoder_ffn_dim=..., decoder_attention_heads=..., encoder_layerdrop=..., decoder_layerdrop=..., use_cache=..., is_encoder_decoder=..., activation_function=..., d_model=..., dropout=..., attention_dropout=..., activation_dropout=..., init_std=..., decoder_start_token_id=..., classifier_dropout=..., scale_embedding=..., gradient_checkpointing=..., pad_token_id=..., bos_token_id=..., eos_token_id=..., max_source_positions=..., max_target_positions=..., num_conv_layers=..., conv_kernel_sizes=..., conv_channels=..., input_feat_per_channel=..., input_channels=..., **kwargs) -> None:
        ...
    
    @property
    def num_attention_heads(self) -> int:
        ...
    
    @property
    def hidden_size(self) -> int:
        ...
    


