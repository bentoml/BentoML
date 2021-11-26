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
    def __init__(self, **kwargs) -> None: ...
    @property
    def is_fast(self) -> bool: ...
    @property
    def vocab_size(self) -> int: ...
    def get_added_vocab(self) -> Dict[str, int]: ...
    def __len__(self): ...
    def num_special_tokens_to_add(self, pair: bool = ...) -> int: ...
    def tokenize(self, text: TextInput, **kwargs) -> List[str]: ...
    def convert_tokens_to_ids(
        self, tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]: ...
    def prepare_for_tokenization(
        self, text: str, is_split_into_words: bool = ..., **kwargs
    ) -> Tuple[str, Dict[str, Any]]: ...
    def get_special_tokens_mask(
        self,
        token_ids_0: List,
        token_ids_1: Optional[List] = ...,
        already_has_special_tokens: bool = ...,
    ) -> List[int]: ...
    @overload
    def convert_ids_to_tokens(
        self, ids: int, skip_special_tokens: bool = ...
    ) -> str: ...
    @overload
    def convert_ids_to_tokens(
        self, ids: List[int], skip_special_tokens: bool = ...
    ) -> List[str]: ...
    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = ...
    ) -> Union[str, List[str]]: ...
    def convert_tokens_to_string(self, tokens: List[str]) -> str: ...
