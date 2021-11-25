

from typing import Any, Mapping, Optional

from transformers import PreTrainedTokenizer, TensorType

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast

logger = ...
GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class GPT2Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.GPT2Model` or a
    :class:`~transformers.TFGPT2Model`. It is used to instantiate a GPT-2 model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the GPT-2 `small <https://huggingface.co/gpt2>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.GPT2Model` or
            :class:`~transformers.TFGPT2Model`.
        n_positions (:obj:`int`, `optional`, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_ctx (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the causal mask (usually same as n_positions).
        n_embd (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (:obj:`int`, `optional`, defaults to None):
            Dimensionality of the inner feed-forward layers. :obj:`None` will set it to 4 times n_embd
        activation_function (:obj:`str`, `optional`, defaults to :obj:`"gelu"`):
            Activation function, to be selected in the list :obj:`["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (:obj:`int`, `optional`, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (:obj:`float`, `optional`, defaults to 1e-5):
            The epsilon to use in the layer normalization layers
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        summary_type (:obj:`string`, `optional`, defaults to :obj:`"cls_index"`):
            Argument used when doing sequence summary, used in the models :class:`~transformers.GPT2DoubleHeadsModel`
            and :class:`~transformers.TFGPT2DoubleHeadsModel`.

            Has to be one of the following options:

                - :obj:`"last"`: Take the last token hidden state (like XLNet).
                - :obj:`"first"`: Take the first token hidden state (like BERT).
                - :obj:`"mean"`: Take the mean of all tokens hidden states.
                - :obj:`"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - :obj:`"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Argument used when doing sequence summary, used in the models :class:`~transformers.GPT2DoubleHeadsModel`
            and :class:`~transformers.TFGPT2DoubleHeadsModel`.

            Whether or not to add a projection after the vector extraction.
        summary_activation (:obj:`str`, `optional`):
            Argument used when doing sequence summary. Used in for the multiple choice head in
            :class:`~transformers.GPT2DoubleHeadsModel`.

            Pass :obj:`"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Argument used when doing sequence summary, used in the models :class:`~transformers.GPT2DoubleHeadsModel`
            and :class:`~transformers.TFGPT2DoubleHeadsModel`.

            Whether the projection outputs should have :obj:`config.num_labels` or :obj:`config.hidden_size` classes.
        summary_first_dropout (:obj:`float`, `optional`, defaults to 0.1):
            Argument used when doing sequence summary, used in the models :class:`~transformers.GPT2DoubleHeadsModel`
            and :class:`~transformers.TFGPT2DoubleHeadsModel`.

            The dropout ratio to be used after the projection and activation.
        scale_attn_weights (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Scale attention weights by dividing by sqrt(hidden_size).
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models).

    Example::

        >>> from transformers import GPT2Model, GPT2Config

        >>> # Initializing a GPT2 configuration
        >>> configuration = GPT2Config()

        >>> # Initializing a model from the configuration
        >>> model = GPT2Model(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(self, vocab_size=..., n_positions=..., n_ctx=..., n_embd=..., n_layer=..., n_head=..., n_inner=..., activation_function=..., resid_pdrop=..., embd_pdrop=..., attn_pdrop=..., layer_norm_epsilon=..., initializer_range=..., summary_type=..., summary_use_proj=..., summary_activation=..., summary_proj_to_labels=..., summary_first_dropout=..., scale_attn_weights=..., gradient_checkpointing=..., use_cache=..., bos_token_id=..., eos_token_id=..., **kwargs) -> None:
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
    


class GPT2OnnxConfig(OnnxConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        ...
    
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        ...
    
    def generate_dummy_inputs(self, tokenizer: PreTrainedTokenizer, batch_size: int = ..., seq_length: int = ..., is_pair: bool = ..., framework: Optional[TensorType] = ...) -> Mapping[str, Any]:
        ...
    


