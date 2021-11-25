

from ...configuration_utils import PretrainedConfig

logger = ...
DETR_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class DetrConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.DetrModel`. It is used to
    instantiate a DETR model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the DETR `facebook/detr-resnet-50
    <https://huggingface.co/facebook/detr-resnet-50>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        num_queries (:obj:`int`, `optional`, defaults to 100):
            Number of object queries, i.e. detection slots. This is the maximal number of objects
            :class:`~transformers.DetrModel` can detect in a single image. For COCO, we recommend 100 queries.
        d_model (:obj:`int`, `optional`, defaults to 256):
            Dimension of the layers.
        encoder_layers (:obj:`int`, `optional`, defaults to 6):
            Number of encoder layers.
        decoder_layers (:obj:`int`, `optional`, defaults to 6):
            Number of decoder layers.
        encoder_attention_heads (:obj:`int`, `optional`, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (:obj:`int`, `optional`, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (:obj:`int`, `optional`, defaults to 2048):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (:obj:`int`, `optional`, defaults to 2048):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        init_std (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        init_xavier_std (:obj:`float`, `optional`, defaults to 1):
            The scaling factor used for the Xavier initialization gain in the HM Attention map module.
        encoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the encoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        decoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the decoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        auxiliary_loss (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"sine"`):
            Type of position embeddings to be used on top of the image features. One of :obj:`"sine"` or
            :obj:`"learned"`.
        backbone (:obj:`str`, `optional`, defaults to :obj:`"resnet50"`):
            Name of convolutional backbone to use. Supports any convolutional backbone from the timm package. For a
            list of all available models, see `this page
            <https://rwightman.github.io/pytorch-image-models/#load-a-pretrained-model>`__.
        dilation (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to replace stride with dilation in the last convolutional block (DC5).
        class_cost (:obj:`float`, `optional`, defaults to 1):
            Relative weight of the classification error in the Hungarian matching cost.
        bbox_cost (:obj:`float`, `optional`, defaults to 5):
            Relative weight of the L1 error of the bounding box coordinates in the Hungarian matching cost.
        giou_cost (:obj:`float`, `optional`, defaults to 2):
            Relative weight of the generalized IoU loss of the bounding box in the Hungarian matching cost.
        mask_loss_coefficient (:obj:`float`, `optional`, defaults to 1):
            Relative weight of the Focal loss in the panoptic segmentation loss.
        dice_loss_coefficient (:obj:`float`, `optional`, defaults to 1):
            Relative weight of the DICE/F-1 loss in the panoptic segmentation loss.
        bbox_loss_coefficient (:obj:`float`, `optional`, defaults to 5):
            Relative weight of the L1 bounding box loss in the object detection loss.
        giou_loss_coefficient (:obj:`float`, `optional`, defaults to 2):
            Relative weight of the generalized IoU loss in the object detection loss.
        eos_coefficient (:obj:`float`, `optional`, defaults to 0.1):
            Relative classification weight of the 'no-object' class in the object detection loss.

    Examples::

        >>> from transformers import DetrModel, DetrConfig

        >>> # Initializing a DETR facebook/detr-resnet-50 style configuration
        >>> configuration = DetrConfig()

        >>> # Initializing a model from the facebook/detr-resnet-50 style configuration
        >>> model = DetrModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(self, num_queries=..., max_position_embeddings=..., encoder_layers=..., encoder_ffn_dim=..., encoder_attention_heads=..., decoder_layers=..., decoder_ffn_dim=..., decoder_attention_heads=..., encoder_layerdrop=..., decoder_layerdrop=..., is_encoder_decoder=..., activation_function=..., d_model=..., dropout=..., attention_dropout=..., activation_dropout=..., init_std=..., init_xavier_std=..., classifier_dropout=..., scale_embedding=..., auxiliary_loss=..., position_embedding_type=..., backbone=..., dilation=..., class_cost=..., bbox_cost=..., giou_cost=..., mask_loss_coefficient=..., dice_loss_coefficient=..., bbox_loss_coefficient=..., giou_loss_coefficient=..., eos_coefficient=..., **kwargs) -> None:
        ...
    
    @property
    def num_attention_heads(self) -> int:
        ...
    
    @property
    def hidden_size(self) -> int:
        ...
    


