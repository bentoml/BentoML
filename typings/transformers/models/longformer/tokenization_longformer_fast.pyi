

from ..roberta.tokenization_roberta_fast import RobertaTokenizerFast
from .tokenization_longformer import LongformerTokenizer

logger = ...
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
class LongformerTokenizerFast(RobertaTokenizerFast):
    r"""
    Construct a "fast" Longformer tokenizer (backed by HuggingFace's `tokenizers` library).

    :class:`~transformers.LongformerTokenizerFast` is identical to :class:`~transformers.RobertaTokenizerFast`. Refer
    to the superclass for usage examples and documentation concerning parameters.
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    slow_tokenizer_class = LongformerTokenizer


