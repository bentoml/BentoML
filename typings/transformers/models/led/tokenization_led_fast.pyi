

from ..bart.tokenization_bart_fast import BartTokenizerFast
from .tokenization_led import LEDTokenizer

logger = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
class LEDTokenizerFast(BartTokenizerFast):
    r"""
    Construct a "fast" LED tokenizer (backed by HuggingFace's `tokenizers` library).

    :class:`~transformers.LEDTokenizerFast` is identical to :class:`~transformers.BartTokenizerFast` and runs
    end-to-end tokenization: punctuation splitting and wordpiece.

    Refer to superclass :class:`~transformers.BartTokenizerFast` for usage examples and documentation concerning
    parameters.
    """
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    slow_tokenizer_class = LEDTokenizer


