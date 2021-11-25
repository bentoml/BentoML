

from typing import Optional, Tuple

from ...file_utils import is_sentencepiece_available
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from .tokenization_reformer import ReformerTokenizer

if is_sentencepiece_available():
    ...
else:
    ReformerTokenizer = ...
logger = ...
SPIECE_UNDERLINE = ...
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
class ReformerTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" Reformer tokenizer (backed by HuggingFace's `tokenizers` library). Based on `Unigram
    <https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models>`__.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizerFast` which contains most of the main
    methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a `.spm` extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sequence token.

            .. note::

                When building a sequence using special tokens, this is not the token that is used for the end of
                sequence. The token used is the :obj:`sep_token`.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        additional_special_tokens (:obj:`List[str]`, `optional`):
            Additional special tokens used by the tokenizer.
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    model_input_names = ...
    slow_tokenizer_class = ...
    def __init__(self, vocab_file=..., tokenizer_file=..., eos_token=..., unk_token=..., additional_special_tokens=..., **kwargs) -> None:
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    


