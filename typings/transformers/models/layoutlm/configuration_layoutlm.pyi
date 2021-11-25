

from ..bert.configuration_bert import BertConfig

logger = ...
LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class LayoutLMConfig(BertConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.LayoutLMModel`. It is used to
    instantiate a LayoutLM model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the LayoutLM `layoutlm-base-uncased
    <https://huggingface.co/microsoft/layoutlm-base-uncased>`__ architecture.

    Configuration objects inherit from :class:`~transformers.BertConfig` and can be used to control the model outputs.
    Read the documentation from :class:`~transformers.BertConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the LayoutLM model. Defines the different tokens that can be represented by the
            `inputs_ids` passed to the forward method of :class:`~transformers.LayoutLMModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
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
            The vocabulary size of the :obj:`token_type_ids` passed into :class:`~transformers.LayoutLMModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        max_2d_position_embeddings (:obj:`int`, `optional`, defaults to 1024):
            The maximum value that the 2D position embedding might ever used. Typically set this to something large
            just in case (e.g., 1024).

    Examples::

        >>> from transformers import LayoutLMModel, LayoutLMConfig

        >>> # Initializing a LayoutLM configuration
        >>> configuration = LayoutLMConfig()

        >>> # Initializing a model from the configuration
        >>> model = LayoutLMModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config

    """
    model_type = ...
    def __init__(self, vocab_size=..., hidden_size=..., num_hidden_layers=..., num_attention_heads=..., intermediate_size=..., hidden_act=..., hidden_dropout_prob=..., attention_probs_dropout_prob=..., max_position_embeddings=..., type_vocab_size=..., initializer_range=..., layer_norm_eps=..., pad_token_id=..., gradient_checkpointing=..., max_2d_position_embeddings=..., **kwargs) -> None:
        ...
    


