

from ..bert.tokenization_bert_fast import BertTokenizerFast
from .tokenization_retribert import RetriBertTokenizer

logger = ...
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
PRETRAINED_INIT_CONFIGURATION = ...
class RetriBertTokenizerFast(BertTokenizerFast):
    r"""
    Construct a "fast" RetriBERT tokenizer (backed by HuggingFace's `tokenizers` library).

    :class:`~transformers.RetriBertTokenizerFast` is identical to :class:`~transformers.BertTokenizerFast` and runs
    end-to-end tokenization: punctuation splitting and wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizerFast` for usage examples and documentation concerning
    parameters.
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    pretrained_init_configuration = ...
    slow_tokenizer_class = RetriBertTokenizer
    model_input_names = ...


