

from ...configuration_utils import PretrainedConfig

TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP = ...
class TapasConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.TapasModel`. It is used to
    instantiate a TAPAS model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the TAPAS `tapas-base-finetuned-sqa`
    architecture. Configuration objects inherit from :class:`~transformers.PreTrainedConfig` and can be used to control
    the model outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Hyperparameters additional to BERT are taken from run_task_main.py and hparam_utils.py of the original
    implementation. Original implementation available at https://github.com/google-research/tapas/tree/master.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the TAPAS model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.TapasModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"swish"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_sizes (:obj:`List[int]`, `optional`, defaults to :obj:`[3, 256, 256, 2, 256, 256, 10]`):
            The vocabulary sizes of the :obj:`token_type_ids` passed when calling :class:`~transformers.TapasModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use gradient checkpointing to save memory at the expense of a slower backward pass.
        positive_label_weight (:obj:`float`, `optional`, defaults to 10.0):
            Weight for positive labels.
        num_aggregation_labels (:obj:`int`, `optional`, defaults to 0):
            The number of aggregation operators to predict.
        aggregation_loss_weight (:obj:`float`, `optional`, defaults to 1.0):
            Importance weight for the aggregation loss.
        use_answer_as_supervision (:obj:`bool`, `optional`):
            Whether to use the answer as the only supervision for aggregation examples.
        answer_loss_importance (:obj:`float`, `optional`, defaults to 1.0):
            Importance weight for the regression loss.
        use_normalized_answer_loss (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to normalize the answer loss by the maximum of the predicted and expected value.
        huber_loss_delta (:obj:`float`, `optional`):
            Delta parameter used to calculate the regression loss.
        temperature (:obj:`float`, `optional`, defaults to 1.0):
            Value used to control (OR change) the skewness of cell logits probabilities.
        aggregation_temperature (:obj:`float`, `optional`, defaults to 1.0):
            Scales aggregation logits to control the skewness of probabilities.
        use_gumbel_for_cells (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to apply Gumbel-Softmax to cell selection.
        use_gumbel_for_aggregation (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to apply Gumbel-Softmax to aggregation selection.
        average_approximation_function (:obj:`string`, `optional`, defaults to :obj:`"ratio"`):
            Method to calculate the expected average of cells in the weak supervision case. One of :obj:`"ratio"`,
            :obj:`"first_order"` or :obj:`"second_order"`.
        cell_selection_preference (:obj:`float`, `optional`):
            Preference for cell selection in ambiguous cases. Only applicable in case of weak supervision for
            aggregation (WTQ, WikiSQL). If the total mass of the aggregation probabilities (excluding the "NONE"
            operator) is higher than this hyperparameter, then aggregation is predicted for an example.
        answer_loss_cutoff (:obj:`float`, `optional`):
            Ignore examples with answer loss larger than cutoff.
        max_num_rows (:obj:`int`, `optional`, defaults to 64):
            Maximum number of rows.
        max_num_columns (:obj:`int`, `optional`, defaults to 32):
            Maximum number of columns.
        average_logits_per_cell (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to average logits per cell.
        select_one_column (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to constrain the model to only select cells from a single column.
        allow_empty_column_selection (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to allow not to select any column.
        init_cell_selection_weights_to_zero (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to initialize cell selection weights to 0 so that the initial probabilities are 50%.
        reset_position_index_per_cell (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to restart position indexes at every cell (i.e. use relative position embeddings).
        disable_per_token_loss (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to disable any (strong or weak) supervision on cells.
        aggregation_labels (:obj:`Dict[int, label]`, `optional`):
            The aggregation labels used to aggregate the results. For example, the WTQ models have the following
            aggregation labels: :obj:`{0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}`
        no_aggregation_label_index (:obj:`int`, `optional`):
            If the aggregation labels are defined and one of these labels represents "No aggregation", this should be
            set to its index. For example, the WTQ models have the "NONE" aggregation label at index 0, so that value
            should be set to 0 for these models.


    Example::

        >>> from transformers import TapasModel, TapasConfig
        >>> # Initializing a default (SQA) Tapas configuration
        >>> configuration = TapasConfig()
        >>> # Initializing a model from the configuration
        >>> model = TapasModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = ...
    def __init__(self, vocab_size=..., hidden_size=..., num_hidden_layers=..., num_attention_heads=..., intermediate_size=..., hidden_act=..., hidden_dropout_prob=..., attention_probs_dropout_prob=..., max_position_embeddings=..., type_vocab_sizes=..., initializer_range=..., layer_norm_eps=..., pad_token_id=..., gradient_checkpointing=..., positive_label_weight=..., num_aggregation_labels=..., aggregation_loss_weight=..., use_answer_as_supervision=..., answer_loss_importance=..., use_normalized_answer_loss=..., huber_loss_delta=..., temperature=..., aggregation_temperature=..., use_gumbel_for_cells=..., use_gumbel_for_aggregation=..., average_approximation_function=..., cell_selection_preference=..., answer_loss_cutoff=..., max_num_rows=..., max_num_columns=..., average_logits_per_cell=..., select_one_column=..., allow_empty_column_selection=..., init_cell_selection_weights_to_zero=..., reset_position_index_per_cell=..., disable_per_token_loss=..., aggregation_labels=..., no_aggregation_label_index=..., **kwargs) -> None:
        ...
    


