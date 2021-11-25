

from ...configuration_utils import PretrainedConfig

logger = ...
HUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class HubertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.HubertModel`. It is used to
    instantiate an Hubert model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Hubert
    `facebook/hubert-base-ls960 <https://huggingface.co/facebook/hubert-base-ls960>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 32):
            Vocabulary size of the Hubert model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.HubertModel`. Vocabulary size of the model.
            Defines the different tokens that can be represented by the `inputs_ids` passed to the forward method of
            :class:`~transformers.HubertModel`.
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
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        feat_extract_norm (:obj:`str`, `optional`, defaults to :obj:`"group"`):
            The norm to be applied to 1D convolutional layers in feature extractor. One of :obj:`"group"` for group
            normalization of only the first 1D convolutional layer or :obj:`"layer"` for layer normalization of all 1D
            convolutional layers.
        feat_extract_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout probabilitiy for all 1D convolutional layers in feature extractor.
        feat_extract_activation (:obj:`str, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the 1D convolutional layers of the feature
            extractor. If string, :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        conv_dim (:obj:`Tuple[int]`, `optional`, defaults to :obj:`(512, 512, 512, 512, 512, 512, 512)`):
            A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the
            feature extractor. The length of `conv_dim` defines the number of 1D convolutional layers.
        conv_stride (:obj:`Tuple[int]`, `optional`, defaults to :obj:`(5, 2, 2, 2, 2, 2, 2)`):
            A tuple of integers defining the stride of each 1D convolutional layer in the feature extractor. The length
            of `conv_stride` defines the number of convolutional layers and has to match the the length of `conv_dim`.
        conv_kernel (:obj:`Tuple[int]`, `optional`, defaults to :obj:`(10, 3, 3, 3, 3, 3, 3)`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the feature extractor. The
            length of `conv_kernel` defines the number of convolutional layers and has to match the the length of
            `conv_dim`.
        conv_bias (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether the 1D convolutional layers have a bias.
        num_conv_pos_embeddings (:obj:`int`, `optional`, defaults to 128):
            Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
            embeddings layer.
        num_conv_pos_embedding_groups (:obj:`int`, `optional`, defaults to 16):
            Number of groups of 1D convolutional positional embeddings layer.
        do_stable_layer_norm (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether do apply `stable` layer norm architecture of the Transformer encoder. ``do_stable_layer_norm is
            True`` corresponds to applying layer norm before the attention layer, whereas ``do_stable_layer_norm is
            False`` corresponds to applying layer norm after the attention layer.
        apply_spec_augment (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to apply *SpecAugment* data augmentation to the outputs of the feature extractor. For reference see
            `SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
            <https://arxiv.org/abs/1904.08779>`__.
        mask_time_prob (:obj:`float`, `optional`, defaults to 0.05):
            Propability of each feature vector along the time axis to be chosen as the start of the vector span to be
            masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature vectors will be
            masked along the time axis. This is only relevant if ``apply_spec_augment is True``.
        mask_time_length (:obj:`int`, `optional`, defaults to 10):
            Length of vector span along the time axis.
        mask_feature_prob (:obj:`float`, `optional`, defaults to 0.0):
            Propability of each feature vector along the feature axis to be chosen as the start of the vector span to
            be masked. Approximately ``mask_time_prob * hidden_size // mask_time_length`` feature vectors will be
            masked along the time axis. This is only relevant if ``apply_spec_augment is True``.
        mask_feature_length (:obj:`int`, `optional`, defaults to 10):
            Length of vector span along the feature axis.
        ctc_loss_reduction (:obj:`str`, `optional`, defaults to :obj:`"sum"`):
            Specifies the reduction to apply to the output of ``torch.nn.CTCLoss``. Only relevant when training an
            instance of :class:`~transformers.HubertForCTC`.
        ctc_zero_infinity (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to zero infinite losses and the associated gradients of ``torch.nn.CTCLoss``. Infinite losses
            mainly occur when the inputs are too short to be aligned to the targets. Only relevant when training an
            instance of :class:`~transformers.HubertForCTC`.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.

    Example::

        >>> from transformers import HubertModel, HubertConfig

        >>> # Initializing a Hubert facebook/hubert-base-ls960 style configuration
        >>> configuration = HubertConfig()

        >>> # Initializing a model from the facebook/hubert-base-ls960 style configuration
        >>> model = HubertModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = ...
    def __init__(self, vocab_size=..., hidden_size=..., num_hidden_layers=..., num_attention_heads=..., intermediate_size=..., hidden_act=..., hidden_dropout=..., activation_dropout=..., attention_dropout=..., feat_proj_dropout=..., final_dropout=..., layerdrop=..., initializer_range=..., layer_norm_eps=..., feat_extract_norm=..., feat_extract_activation=..., conv_dim=..., conv_stride=..., conv_kernel=..., conv_bias=..., num_conv_pos_embeddings=..., num_conv_pos_embedding_groups=..., do_stable_layer_norm=..., apply_spec_augment=..., mask_time_prob=..., mask_time_length=..., mask_feature_prob=..., mask_feature_length=..., ctc_loss_reduction=..., ctc_zero_infinity=..., gradient_checkpointing=..., pad_token_id=..., bos_token_id=..., eos_token_id=..., **kwargs) -> None:
        ...
    


