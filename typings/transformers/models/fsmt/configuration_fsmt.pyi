

from ...configuration_utils import PretrainedConfig

logger = ...
FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class DecoderConfig(PretrainedConfig):
    r"""
    Configuration class for FSMT's decoder specific things. note: this is a private helper class
    """
    model_type = ...
    def __init__(self, vocab_size=..., bos_token_id=...) -> None:
        ...
    


class FSMTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.FSMTModel`. It is used to
    instantiate a FSMT model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        langs (:obj:`List[str]`):
            A list with source language and target_language (e.g., ['en', 'ru']).
        src_vocab_size (:obj:`int`):
            Vocabulary size of the encoder. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed to the forward method in the encoder.
        tgt_vocab_size (:obj:`int`):
            Vocabulary size of the decoder. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed to the forward method in the decoder.
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
        activation_function (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_embedding (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Scale embeddings by diving by sqrt(d_model).
        bos_token_id (:obj:`int`, `optional`, defaults to 0)
            Beginning of stream token id.
        pad_token_id (:obj:`int`, `optional`, defaults to 1)
            Padding token id.
        eos_token_id (:obj:`int`, `optional`, defaults to 2)
            End of stream token id.
        decoder_start_token_id (:obj:`int`, `optional`):
            This model starts decoding with :obj:`eos_token_id`
        encoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            Google "layerdrop arxiv", as its not explainable in one line.
        decoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            Google "layerdrop arxiv", as its not explainable in one line.
        is_encoder_decoder (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether this is an encoder/decoder model.
        tie_word_embeddings (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to tie input and output embeddings.
        num_beams (:obj:`int`, `optional`, defaults to 5)
            Number of beams for beam search that will be used by default in the :obj:`generate` method of the model. 1
            means no beam search.
        length_penalty (:obj:`float`, `optional`, defaults to 1)
            Exponential penalty to the length that will be used by default in the :obj:`generate` method of the model.
        early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`)
            Flag that will be used by default in the :obj:`generate` method of the model. Whether to stop the beam
            search when at least ``num_beams`` sentences are finished per batch or not.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        forced_eos_token_id (:obj:`int`, `optional`, defaults to 2):
            The id of the token to force as the last generated token when :obj:`max_length` is reached. Usually set to
            :obj:`eos_token_id`.

        Examples::

            >>> from transformers import FSMTConfig, FSMTModel

            >>> config = FSMTConfig.from_pretrained('facebook/wmt19-en-ru')
            >>> model = FSMTModel(config)

    """
    model_type = ...
    def __init__(self, langs=..., src_vocab_size=..., tgt_vocab_size=..., activation_function=..., d_model=..., max_length=..., max_position_embeddings=..., encoder_ffn_dim=..., encoder_layers=..., encoder_attention_heads=..., encoder_layerdrop=..., decoder_ffn_dim=..., decoder_layers=..., decoder_attention_heads=..., decoder_layerdrop=..., attention_dropout=..., dropout=..., activation_dropout=..., init_std=..., decoder_start_token_id=..., is_encoder_decoder=..., scale_embedding=..., tie_word_embeddings=..., num_beams=..., length_penalty=..., early_stopping=..., use_cache=..., pad_token_id=..., bos_token_id=..., eos_token_id=..., forced_eos_token_id=..., **common_kwargs) -> None:
        ...
    
    @property
    def num_attention_heads(self) -> int:
        ...
    
    @property
    def hidden_size(self) -> int:
        ...
    
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default `to_dict()` from `PretrainedConfig`.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        ...
    


