

from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer

if TYPE_CHECKING:
    ...
logger = ...
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    ...

def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    ...

class GPT2Tokenizer(PreTrainedTokenizer):
    """
    Construct a GPT-2 tokenizer. Based on byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ::

        >>> from transformers import GPT2Tokenizer
        >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        >>> tokenizer("Hello world")['input_ids']
        [15496, 995]
        >>> tokenizer(" Hello world")['input_ids']
        [18435, 995]

    You can get around that behavior by passing ``add_prefix_space=True`` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    .. note::

        When used with ``is_split_into_words=True``, this tokenizer will add a space before each word (even the first
        one).

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        merges_file (:obj:`str`):
            Path to the merges file.
        errors (:obj:`str`, `optional`, defaults to :obj:`"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See `bytes.decode
            <https://docs.python.org/3/library/stdtypes.html#bytes.decode>`__ for more information.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`<|endoftext|>`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`str`, `optional`, defaults to :obj:`<|endoftext|>`):
            The beginning of sequence token.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`<|endoftext|>`):
            The end of sequence token.
        add_prefix_space (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (GPT2 tokenizer detect beginning of words by the preceding space).
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    model_input_names = ...
    def __init__(self, vocab_file, merges_file, errors=..., unk_token=..., bos_token=..., eos_token=..., add_prefix_space=..., **kwargs) -> None:
        ...
    
    @property
    def vocab_size(self):
        ...
    
    def get_vocab(self):
        ...
    
    def bpe(self, token):
        ...
    
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    
    def prepare_for_tokenization(self, text, is_split_into_words=..., **kwargs):
        ...
    


