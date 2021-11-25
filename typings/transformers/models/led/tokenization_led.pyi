

from ..bart.tokenization_bart import BartTokenizer

logger = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
class LEDTokenizer(BartTokenizer):
    """
    Construct a LED tokenizer.

    :class:`~transformers.LEDTokenizer` is identical to :class:`~transformers.BartTokenizer` and runs end-to-end
    tokenization: punctuation splitting and wordpiece.

    Refer to superclass :class:`~transformers.BartTokenizer` for usage examples and documentation concerning
    parameters.
    """
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...


