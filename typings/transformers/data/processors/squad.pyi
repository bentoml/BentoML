

from ...file_utils import is_tf_available, is_torch_available
from ...tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from .utils import DataProcessor

MULTI_SEP_TOKENS_TOKENIZERS_SET = ...
if is_torch_available():
    ...
if is_tf_available():
    ...
logger = ...
def squad_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training):
    ...

def squad_convert_example_to_features_init(tokenizer_for_convert: PreTrainedTokenizerBase):
    ...

def squad_convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training, padding_strategy=..., return_dataset=..., threads=..., tqdm_enabled=...):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model. It is
    model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        padding_strategy: Default to "max_length". Which padding strategy to use
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset, if 'tf': returns a tf.data.Dataset
        threads: multiple processing threads.


    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """
    ...

class SquadProcessor(DataProcessor):
    """
    Processor for the SQuAD data set. overridden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and
    version 2.0 of SQuAD, respectively.
    """
    train_file = ...
    dev_file = ...
    def get_examples_from_dataset(self, dataset, evaluate=...):
        """
        Creates a list of :class:`~transformers.data.processors.squad.SquadExample` using a TFDS dataset.

        Args:
            dataset: The tfds dataset loaded from `tensorflow_datasets.load("squad")`
            evaluate: Boolean specifying if in evaluation mode or in training mode

        Returns:
            List of SquadExample

        Examples::

            >>> import tensorflow_datasets as tfds
            >>> dataset = tfds.load("squad")

            >>> training_examples = get_examples_from_dataset(dataset, evaluate=False)
            >>> evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
        """
        ...
    
    def get_train_examples(self, data_dir, filename=...):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """
        ...
    
    def get_dev_examples(self, data_dir, filename=...):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `dev-v1.1.json` and `dev-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        ...
    


class SquadV1Processor(SquadProcessor):
    train_file = ...
    dev_file = ...


class SquadV2Processor(SquadProcessor):
    train_file = ...
    dev_file = ...


class SquadExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """
    def __init__(self, qas_id, question_text, context_text, answer_text, start_position_character, title, answers=..., is_impossible=...) -> None:
        ...
    


class SquadFeatures:
    """
    Single squad example features to be fed to a model. Those features are model-specific and can be crafted from
    :class:`~transformers.data.processors.squad.SquadExample` using the
    :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
        encoding: optionally store the BatchEncoding with the fast-tokenizer alignment methods.
    """
    def __init__(self, input_ids, attention_mask, token_type_ids, cls_index, p_mask, example_index, unique_id, paragraph_len, token_is_max_context, tokens, token_to_orig_map, start_position, end_position, is_impossible, qas_id: str = ..., encoding: BatchEncoding = ...) -> None:
        ...
    


class SquadResult:
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """
    def __init__(self, unique_id, start_logits, end_logits, start_top_index=..., end_top_index=..., cls_logits=...) -> None:
        ...
    


