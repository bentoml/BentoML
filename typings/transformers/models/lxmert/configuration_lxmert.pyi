

from ...configuration_utils import PretrainedConfig

logger = ...
LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class LxmertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.LxmertModel` or a
    :class:`~transformers.TFLxmertModel`. It is used to instantiate a LXMERT model according to the specified
    arguments, defining the model architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the LXMERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.LxmertModel` or
            :class:`~transformers.TFLxmertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        r_layers (:obj:`int`, `optional`, defaults to 5):
            Number of hidden layers in the Transformer visual encoder.
        l_layers (:obj:`int`, `optional`, defaults to 9):
            Number of hidden layers in the Transformer language encoder.
        x_layers (:obj:`int`, `optional`, defaults to 5):
            Number of hidden layers in the Transformer cross modality encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 5):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
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
            The vocabulary size of the `token_type_ids` passed into :class:`~transformers.BertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        visual_feat_dim (:obj:`int`, `optional`, defaults to 2048):
            This represents the last dimension of the pooled-object features used as input for the model, representing
            the size of each object feature itself.
        visual_pos_dim (:obj:`int`, `optional`, defaults to 4):
            This represents the number of spacial features that are mixed into the visual features. The default is set
            to 4 because most commonly this will represent the location of a bounding box. i.e., (x, y, width, height)
        visual_loss_normalizer (:obj:`float`, `optional`, defaults to 1/15):
            This represents the scaling factor in which each visual loss is multiplied by if during pretraining, one
            decided to train with multiple vision-based loss objectives.
        num_qa_labels (:obj:`int`, `optional`, defaults to 9500):
            This represents the total number of different question answering (QA) labels there are. If using more than
            one dataset with QA, the user will need to account for the total number of labels that all of the datasets
            have in total.
        num_object_labels (:obj:`int`, `optional`, defaults to 1600):
            This represents the total number of semantically unique objects that lxmert will be able to classify a
            pooled-object feature as belonging too.
        num_attr_labels (:obj:`int`, `optional`, defaults to 400):
            This represents the total number of semantically unique attributes that lxmert will be able to classify a
            pooled-object feature as possessing.
        task_matched (:obj:`bool`, `optional`, defaults to :obj:`True`):
            This task is used for sentence-image matching. If the sentence correctly describes the image the label will
            be 1. If the sentence does not correctly describe the image, the label will be 0.
        task_mask_lm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to add masked language modeling (as used in pretraining models such as BERT) to the loss
            objective.
        task_obj_predict (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to add object prediction, attribute prediction and feature regression to the loss objective.
        task_qa (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to add the question-answering loss to the objective
        visual_obj_loss (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to calculate the object-prediction loss objective
        visual_attr_loss (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to calculate the attribute-prediction loss objective
        visual_feat_loss (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to calculate the feature-regression loss objective
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should return the attentions from the vision, language, and cross-modality layers
            should be returned.
        output_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should return the hidden states from the vision, language, and cross-modality
            layers should be returned.
    """
    model_type = ...
    def __init__(self, vocab_size=..., hidden_size=..., num_attention_heads=..., num_labels=..., num_qa_labels=..., num_object_labels=..., num_attr_labels=..., intermediate_size=..., hidden_act=..., hidden_dropout_prob=..., attention_probs_dropout_prob=..., max_position_embeddings=..., type_vocab_size=..., initializer_range=..., layer_norm_eps=..., pad_token_id=..., l_layers=..., x_layers=..., r_layers=..., visual_feat_dim=..., visual_pos_dim=..., visual_loss_normalizer=..., task_matched=..., task_mask_lm=..., task_obj_predict=..., task_qa=..., visual_obj_loss=..., visual_attr_loss=..., visual_feat_loss=..., output_attentions=..., output_hidden_states=..., **kwargs) -> None:
        ...
    


