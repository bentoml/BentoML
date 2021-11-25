

from typing import Dict, List, Optional, Union

from tokenizers import Tokenizer as TokenizerFast
from tokenizers.decoders import Decoder as DecoderFast

from .file_utils import PaddingStrategy, add_end_docstrings
from .tokenization_utils import PreTrainedTokenizer
from .tokenization_utils_base import (
    INIT_TOKENIZER_DOCSTRING,
    PreTrainedTokenizerBase,
    TruncationStrategy,
)

logger = ...
TOKENIZER_FILE = ...
SPECIAL_TOKENS_MAP_FILE = ...
TOKENIZER_CONFIG_FILE = ...
ADDED_TOKENS_FILE = ...
MODEL_TO_TRAINER_MAPPING = ...
@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class PreTrainedTokenizerFast(PreTrainedTokenizerBase):
    """
    Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).

    Inherits from :class:`~transformers.tokenization_utils_base.PreTrainedTokenizerBase`.

    Handles all the shared methods for tokenization and special tokens, as well as methods for
    downloading/caching/loading pretrained tokenizers, as well as adding tokens to the vocabulary.

    This class also contains the added tokens in a unified way on top of all tokenizers so we don't have to handle the
    specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).
    """
    slow_tokenizer_class: PreTrainedTokenizer = ...
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @property
    def is_fast(self) -> bool:
        ...
    
    @property
    def vocab_size(self) -> int:
        """
        :obj:`int`: Size of the base vocabulary (without the added tokens).
        """
        ...
    
    def get_vocab(self) -> Dict[str, int]:
        ...
    
    @property
    def vocab(self) -> Dict[str, int]:
        ...
    
    def get_added_vocab(self) -> Dict[str, int]:
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index.

        Returns:
            :obj:`Dict[str, int]`: The added tokens.
        """
        ...
    
    def __len__(self) -> int:
        """
        Size of the full vocabulary with the added tokens.
        """
        ...
    
    @property
    def backend_tokenizer(self) -> TokenizerFast:
        """
        :obj:`tokenizers.implementations.BaseTokenizer`: The Rust tokenizer used as a backend.
        """
        ...
    
    @property
    def decoder(self) -> DecoderFast:
        """
        :obj:`tokenizers.decoders.Decoder`: The Rust decoder for this tokenizer.
        """
        ...
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (:obj:`str` or :obj:`List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            :obj:`int` or :obj:`List[int]`: The token id or list of token ids.
        """
        ...
    
    def num_special_tokens_to_add(self, pair: bool = ...) -> int:
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        .. note::
            This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not
            put this inside your training loop.

        Args:
            pair (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether the number of added tokens should be computed in the case of a sequence pair or a single
                sequence.

        Returns:
            :obj:`int`: Number of special tokens added to sequences.
        """
        ...
    
    def convert_ids_to_tokens(self, ids: Union[int, List[int]], skip_special_tokens: bool = ...) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (:obj:`int` or :obj:`List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            :obj:`str` or :obj:`List[str]`: The decoded token(s).
        """
        ...
    
    def tokenize(self, text: str, pair: Optional[str] = ..., add_special_tokens: bool = ..., **kwargs) -> List[str]:
        ...
    
    def set_truncation_and_padding(self, padding_strategy: PaddingStrategy, truncation_strategy: TruncationStrategy, max_length: int, stride: int, pad_to_multiple_of: Optional[int]):
        """
        Define the truncation and the padding strategies for fast tokenizers (provided by HuggingFace tokenizers
        library) and restore the tokenizer settings afterwards.

        The provided tokenizer has no padding / truncation strategy before the managed section. If your tokenizer set a
        padding / truncation strategy before, then it will be reset to no padding / truncation when exiting the managed
        section.

        Args:
            padding_strategy (:class:`~transformers.file_utils.PaddingStrategy`):
                The kind of padding that will be applied to the input
            truncation_strategy (:class:`~transformers.tokenization_utils_base.TruncationStrategy`):
                The kind of truncation that will be applied to the input
            max_length (:obj:`int`):
                The maximum size of a sequence.
            stride (:obj:`int`):
                The stride to use when handling overflow.
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        """
        ...
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        ...
    
    def train_new_from_iterator(self, text_iterator, vocab_size, new_special_tokens=..., special_tokens_map=..., **kwargs):
        """
        Trains a tokenizer on a new corpus with the same defaults (in terms of special tokens or tokenization pipeline)
        as the current one.

        Args:
            text_iterator (generator of :obj:`List[str]`):
                The training corpus. Should be a generator of batches of texts, for instance a list of lists of texts
                if you have everything in memory.
            vocab_size (obj:`int`):
                The size of the vocabulary you want for your tokenizer.
            new_special_tokens (list of :obj:`str` or :obj:`AddedToken`, `optional`):
                A list of new special tokens to add to the tokenizer you are training.
            special_tokens_map (:obj:`Dict[str, str]`, `optional`):
                If you want to rename some of the special tokens this tokenizer uses, pass along a mapping old special
                token name to new special token name in this argument.
            kwargs:
                Additional keyword arguments passed along to the trainer from the 🤗 Tokenizers library.

        Returns:
            :class:`~transformers.PreTrainedTokenizerFast`: A new tokenizer of the same type as the original one,
            trained on :obj:`text_iterator`.

        """
        ...
    


