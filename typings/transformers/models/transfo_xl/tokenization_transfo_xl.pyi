

from typing import List, Optional, Tuple

from ...file_utils import is_torch_available, torch_only_method
from ...tokenization_utils import PreTrainedTokenizer

if is_torch_available():
    ...
logger = ...
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
PRETRAINED_CORPUS_ARCHIVE_MAP = ...
CORPUS_NAME = ...
MATCH_NUMBERS = ...
DETOKENIZE_NUMBERS = ...
def tokenize_numbers(text_array: List[str]) -> List[str]:
    """
    Splits large comma-separated numbers and floating point values. This is done by replacing commas with ' @,@ ' and
    dots with ' @.@ '.

    Args:
        text_array: An already tokenized text as list.

    Returns:
        A list of strings with tokenized numbers.

    Example::
        >>> tokenize_numbers(["$", "5,000", "1.73", "m"])
        ["$", "5", "@,@", "000", "1", "@.@", "73", "m"]
    """
    ...

def detokenize_numbers(text: str) -> str:
    """
    Inverts the operation of `tokenize_numbers`. This is replacing ' @,@ ' and ' @.@' by ',' and '.'.

    Args:
        text: A string where the number should be detokenized.

    Returns:
        A detokenized string.

    Example::
        >>> detokenize_numbers("$ 5 @,@ 000 1 @.@ 73 m")
        "$ 5,000 1.73 m"
    """
    ...

class TransfoXLTokenizer(PreTrainedTokenizer):
    """
    Construct a Transformer-XL tokenizer adapted from Vocab class in `the original code
    <https://github.com/kimiyoung/transformer-xl>`__. The Transformer-XL tokenizer is a word-level tokenizer (no
    sub-word tokenization).

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        special (:obj:`List[str]`, `optional`):
            A list of special tokens (to be treated by the original implementation of this tokenizer).
        min_freq (:obj:`int`, `optional`, defaults to 0):
            The minimum number of times a token has to be present in order to be kept in the vocabulary (otherwise it
            will be mapped to :obj:`unk_token`).
        max_size (:obj:`int`, `optional`):
            The maximum size of the vocabulary. If left unset, it will default to the size of the vocabulary found
            after excluding the tokens according to the :obj:`min_freq` rule.
        lower_case (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to lowercase the input when tokenizing.
        delimiter (:obj:`str`, `optional`):
            The delimiter used between tokens.
        vocab_file (:obj:`str`, `optional`):
            File containing the vocabulary (from the original implementation).
        pretrained_vocab_file (:obj:`str`, `optional`):
            File containing the vocabulary as saved with the :obj:`save_pretrained()` method.
        never_split (:obj:`List[str]`, `optional`):
            List of tokens that should never be split. If no list is specified, will simply use the existing special
            tokens.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"<eos>"`):
            The end of sequence token.
        additional_special_tokens (:obj:`List[str]`, `optional`, defaults to :obj:`["<formula>"]`):
            A list of additional special tokens (for the HuggingFace functionality).
        language (:obj:`str`, `optional`, defaults to :obj:`"en"`):
            The language of this tokenizer (used for mose preprocessing).
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    model_input_names = ...
    def __init__(self, special=..., min_freq=..., max_size=..., lower_case=..., delimiter=..., vocab_file=..., pretrained_vocab_file: str = ..., never_split=..., unk_token=..., eos_token=..., additional_special_tokens=..., language=..., **kwargs) -> None:
        ...
    
    @property
    def do_lower_case(self):
        ...
    
    def count_file(self, path, verbose=..., add_eos=...):
        ...
    
    def count_sents(self, sents, verbose=...):
        """
        sents : a list of sentences, each a list of tokenized symbols
        """
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    
    def build_vocab(self):
        ...
    
    @torch_only_method
    def encode_file(self, path, ordered=..., verbose=..., add_eos=..., add_double_eos=...):
        ...
    
    @torch_only_method
    def encode_sents(self, sents, ordered=..., verbose=...):
        ...
    
    def add_special(self, sym):
        ...
    
    def add_symbol(self, sym):
        ...
    
    def move_added_token(self, token: str, target_idx: int):
        """
        Moves an added token to a specific position in the vocab. This method should be used when resizing an embedding
        layer other than the last one in the `AdaptiveEmbedding` in order to move the token in the tokenizer from the
        default position (at the very end) to the desired one.

        Args:
            token: The token to move to a specific position in the vocab.
            target_idx: The position where the token should be moved to.
        """
        ...
    
    def moses_punct_norm(self, text):
        ...
    
    def moses_tokenize(self, text):
        ...
    
    def moses_pipeline(self, text: str) -> List[str]:
        """
        Does basic tokenization using :class:`sacremoses.MosesPunctNormalizer` and :class:`sacremoses.MosesTokenizer`
        with `aggressive_dash_splits=True` (see :func:`sacremoses.tokenize.MosesTokenizer.tokenize`). Additionally,
        large comma-separated numbers and floating point values are split. E.g. "23,000 people are 1.80m tall" -> "23
        @,@ 000 people are 1 @.@ 80m tall"

        Args:
            text: Text to be tokenize

        Returns:
            A list of tokenized string

        Example::
            >>> tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
            >>> tokenizer.moses_pipeline("23,000 people are 1.80 m tall")
            ['23', '@,@', '000', 'people', 'are', '1', '@.@', '80', 'm', 'tall']
        """
        ...
    
    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (string) in a single string. Additionally, the split numbers are converted back
        into it's original form.
        """
        ...
    
    @torch_only_method
    def convert_to_tensor(self, symbols):
        ...
    
    @property
    def vocab_size(self):
        ...
    
    def get_vocab(self):
        ...
    


class LMOrderedIterator:
    def __init__(self, data, bsz, bptt, device=..., ext_len=...) -> None:
        """
        data -- LongTensor -- the LongTensor is strictly ordered
        """
        ...
    
    def get_batch(self, i, bptt=...):
        ...
    
    def get_fixlen_iter(self, start=...):
        ...
    
    def get_varlen_iter(self, start=..., std=..., min_len=..., max_deviation=...):
        ...
    
    def __iter__(self):
        ...
    


class LMShuffledIterator:
    def __init__(self, data, bsz, bptt, device=..., ext_len=..., shuffle=...) -> None:
        """
        data -- list[LongTensor] -- there is no order among the LongTensors
        """
        ...
    
    def get_sent_stream(self):
        ...
    
    @torch_only_method
    def stream_iterator(self, sent_stream):
        ...
    
    def __iter__(self):
        ...
    


class LMMultiFileIterator(LMShuffledIterator):
    def __init__(self, paths, vocab, bsz, bptt, device=..., ext_len=..., shuffle=...) -> None:
        ...
    
    def get_sent_stream(self, path):
        ...
    
    def __iter__(self):
        ...
    


class TransfoXLCorpus:
    @classmethod
    @torch_only_method
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=..., *inputs, **kwargs):
        """
        Instantiate a pre-processed corpus.
        """
        ...
    
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    def build_corpus(self, path, dataset):
        ...
    
    def get_iterator(self, split, *args, **kwargs):
        ...
    


@torch_only_method
def get_lm_corpus(datadir, dataset):
    ...

