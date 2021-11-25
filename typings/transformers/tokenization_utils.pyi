

from typing import Any, Dict, List, Optional, Tuple, Union, overload

from .file_utils import add_end_docstrings
from .tokenization_utils_base import (
    INIT_TOKENIZER_DOCSTRING,
    PreTrainedTokenizerBase,
    TextInput,
)

logger = ...
SPECIAL_TOKENS_MAP_FILE = ...
ADDED_TOKENS_FILE = ...
TOKENIZER_CONFIG_FILE = ...
@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class PreTrainedTokenizer(PreTrainedTokenizerBase):
    """
    Base class for all slow tokenizers.

    Inherits from :class:`~transformers.tokenization_utils_base.PreTrainedTokenizerBase`.

    Handle all the shared methods for tokenization and special tokens as well as methods downloading/caching/loading
    pretrained tokenizers as well as adding tokens to the vocabulary.

    This class also contain the added tokens in a unified way on top of all tokenizers so we don't have to handle the
    specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).
    """
    def __init__(self, **kwargs) -> None:
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
    
    def get_added_vocab(self) -> Dict[str, int]:
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index.

        Returns:
            :obj:`Dict[str, int]`: The added tokens.
        """
        ...
    
    def __len__(self):
        """
        Size of the full vocabulary with the added tokens.
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
    
    def tokenize(self, text: TextInput, **kwargs) -> List[str]:
        """
        Converts a string in a sequence of tokens, using the tokenizer.

        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
        (BPE/SentencePieces/WordPieces). Takes care of added tokens.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.
            **kwargs (additional keyword arguments):
                Passed along to the model-specific ``prepare_for_tokenization`` preprocessing method.

        Returns:
            :obj:`List[str]`: The list of tokens.
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
    
    def prepare_for_tokenization(self, text: str, is_split_into_words: bool = ..., **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Performs any necessary transformations before tokenization.

        This method should pop the arguments from kwargs and return the remaining :obj:`kwargs` as well. We test the
        :obj:`kwargs` at the end of the encoding process to be sure all the arguments have been used.

        Args:
            text (:obj:`str`):
                The text to prepare.
            is_split_into_words (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to :obj:`True`, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
            kwargs:
                Keyword arguments to use for the tokenization.

        Returns:
            :obj:`Tuple[str, Dict[str, Any]]`: The prepared text and the unused kwargs.
        """
        ...
    
    def get_special_tokens_mask(self, token_ids_0: List, token_ids_1: Optional[List] = ..., already_has_special_tokens: bool = ...) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (:obj:`List[int]`, `optional`):
                List of ids of the second sequence.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        ...
    
    @overload
    def convert_ids_to_tokens(self, ids: int, skip_special_tokens: bool = ...) -> str:
        ...
    
    @overload
    def convert_ids_to_tokens(self, ids: List[int], skip_special_tokens: bool = ...) -> List[str]:
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
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        ...
    


