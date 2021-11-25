

from typing import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

logger = ...
DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class DistilBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.DistilBertModel` or a
    :class:`~transformers.TFDistilBertModel`. It is used to instantiate a DistilBERT model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the DistilBERT `distilbert-base-uncased
    <https://huggingface.co/distilbert-base-uncased>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the DistilBERT model. Defines the number of different tokens that can be represented by
            the :obj:`inputs_ids` passed when calling :class:`~transformers.DistilBertModel` or
            :class:`~transformers.TFDistilBertModel`.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        sinusoidal_pos_embds (:obj:`boolean`, `optional`, defaults to :obj:`False`):
            Whether to use sinusoidal positional embeddings.
        n_layers (:obj:`int`, `optional`, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        n_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        dim (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        hidden_dim (:obj:`int`, `optional`, defaults to 3072):
            The size of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        activation (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        qa_dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilities used in the question answering model
            :class:`~transformers.DistilBertForQuestionAnswering`.
        seq_classif_dropout (:obj:`float`, `optional`, defaults to 0.2):
            The dropout probabilities used in the sequence classification and the multiple choice model
            :class:`~transformers.DistilBertForSequenceClassification`.

    Examples::

        >>> from transformers import DistilBertModel, DistilBertConfig

        >>> # Initializing a DistilBERT configuration
        >>> configuration = DistilBertConfig()

        >>> # Initializing a model from the configuration
        >>> model = DistilBertModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = ...
    def __init__(self, vocab_size=..., max_position_embeddings=..., sinusoidal_pos_embds=..., n_layers=..., n_heads=..., dim=..., hidden_dim=..., dropout=..., attention_dropout=..., activation=..., initializer_range=..., qa_dropout=..., seq_classif_dropout=..., pad_token_id=..., **kwargs) -> None:
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
    


class DistilBertOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        ...
    
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        ...
    


