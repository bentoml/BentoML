

from typing import List, Union

from ...configuration_utils import PretrainedConfig

logger = ...
LED_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class LEDConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.LEDModel`. It is used to
    instantiate an LED model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the LED `allenai/led-base-16384
    <https://huggingface.co/allenai/led-base-16384>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50265):
            Vocabulary size of the LED model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.LEDModel` or :class:`~transformers.TFLEDModel`.
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
        max_encoder_position_embeddings (:obj:`int`, `optional`, defaults to 16384):
            The maximum sequence length that the encoder might ever be used with.
        max_decoder_position_embeddings (:obj:`int`, `optional`, defaults to 16384):
            The maximum sequence length that the decoder might ever be used with.
        init_std (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        encoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the encoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        decoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the decoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models)
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.

        Example::

        >>> from transformers import LEDModel, LEDConfig

        >>> # Initializing a LED allenai/led-base-16384 style configuration
        >>> configuration = LEDConfig()

        >>> # Initializing a model from the allenai/led-base-16384 style configuration
        >>> model = LEDModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = ...
    def __init__(self, vocab_size=..., max_encoder_position_embeddings=..., max_decoder_position_embeddings=..., encoder_layers=..., encoder_ffn_dim=..., encoder_attention_heads=..., decoder_layers=..., decoder_ffn_dim=..., decoder_attention_heads=..., encoder_layerdrop=..., decoder_layerdrop=..., use_cache=..., is_encoder_decoder=..., activation_function=..., d_model=..., dropout=..., attention_dropout=..., activation_dropout=..., init_std=..., decoder_start_token_id=..., classifier_dropout=..., pad_token_id=..., bos_token_id=..., eos_token_id=..., gradient_checkpointing=..., attention_window: Union[List[int], int] = ..., **kwargs) -> None:
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
    
    @property
    def initializer_range(self) -> float:
        ...
    


