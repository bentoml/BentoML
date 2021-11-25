

from ...configuration_utils import PretrainedConfig

logger = ...
ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class ElectraConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.ElectraModel` or a
    :class:`~transformers.TFElectraModel`. It is used to instantiate a ELECTRA model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the ELECTRA `google/electra-small-discriminator
    <https://huggingface.co/google/electra-small-discriminator>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the ELECTRA model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.ElectraModel` or
            :class:`~transformers.TFElectraModel`.
        embedding_size (:obj:`int`, `optional`, defaults to 128):
            Dimensionality of the encoder layers and the pooler layer.
        hidden_size (:obj:`int`, `optional`, defaults to 256):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 4):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
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
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.ElectraModel` or
            :class:`~transformers.TFElectraModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        summary_type (:obj:`str`, `optional`, defaults to :obj:`"first"`):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Has to be one of the following options:

                - :obj:`"last"`: Take the last token hidden state (like XLNet).
                - :obj:`"first"`: Take the first token hidden state (like BERT).
                - :obj:`"mean"`: Take the mean of all tokens hidden states.
                - :obj:`"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - :obj:`"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Whether or not to add a projection after the vector extraction.
        summary_activation (:obj:`str`, `optional`):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Pass :obj:`"gelu"` for a gelu activation to the output, any other value will result in no activation.
        summary_last_dropout (:obj:`float`, `optional`, defaults to 0.0):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            The dropout ratio to be used after the projection and activation.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"absolute"`):
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.

    Examples::

        >>> from transformers import ElectraModel, ElectraConfig

        >>> # Initializing a ELECTRA electra-base-uncased style configuration
        >>> configuration = ElectraConfig()

        >>> # Initializing a model from the electra-base-uncased style configuration
        >>> model = ElectraModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = ...
    def __init__(self, vocab_size=..., embedding_size=..., hidden_size=..., num_hidden_layers=..., num_attention_heads=..., intermediate_size=..., hidden_act=..., hidden_dropout_prob=..., attention_probs_dropout_prob=..., max_position_embeddings=..., type_vocab_size=..., initializer_range=..., layer_norm_eps=..., summary_type=..., summary_use_proj=..., summary_activation=..., summary_last_dropout=..., pad_token_id=..., position_embedding_type=..., **kwargs) -> None:
        ...
    


