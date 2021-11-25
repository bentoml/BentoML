

from typing import TYPE_CHECKING, Optional, Tuple

from ...tokenization_utils_fast import PreTrainedTokenizerFast

if TYPE_CHECKING:
    ...
logger = ...
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
class GPT2TokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" GPT-2 tokenizer (backed by HuggingFace's `tokenizers` library). Based on byte-level
    Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ::

        >>> from transformers import GPT2TokenizerFast
        >>> tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        >>> tokenizer("Hello world")['input_ids']
        [15496, 995]
        >>> tokenizer(" Hello world")['input_ids']
        [18435, 995]

    You can get around that behavior by passing ``add_prefix_space=True`` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    .. note::

        When used with ``is_split_into_words=True``, this tokenizer needs to be instantiated with
        ``add_prefix_space=True``.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizerFast` which contains most of the main
    methods. Users should refer to this superclass for more information regarding those methods.

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
        trim_offsets (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the post-processing step should trim offsets to avoid including whitespaces.
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    model_input_names = ...
    slow_tokenizer_class = ...
    def __init__(self, vocab_file=..., merges_file=..., tokenizer_file=..., unk_token=..., bos_token=..., eos_token=..., add_prefix_space=..., **kwargs) -> None:
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    


