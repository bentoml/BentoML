

from ..roberta.tokenization_roberta_fast import RobertaTokenizerFast
from .tokenization_bart import BartTokenizer

logger = ...
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
class BartTokenizerFast(RobertaTokenizerFast):
    r"""
    Construct a "fast" BART tokenizer (backed by HuggingFace's `tokenizers` library).

    :class:`~transformers.BartTokenizerFast` is identical to :class:`~transformers.RobertaTokenizerFast`. Refer to
    superclass :class:`~transformers.RobertaTokenizerFast` for usage examples and documentation concerning the
    initialization parameters and other methods.
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    slow_tokenizer_class = BartTokenizer


