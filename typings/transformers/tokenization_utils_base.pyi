import os
from collections import UserDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import numpy as np
from tokenizers import AddedToken
from tokenizers import Encoding as EncodingFast
from .file_utils import (
    ExplicitEnum,
    PaddingStrategy,
    PushToHubMixin,
    TensorType,
    add_end_docstrings,
    torch_required,
)

@dataclass(frozen=True, eq=True)
class AddedToken:
    content: str = ...
    single_word: bool = ...
    lstrip: bool = ...
    rstrip: bool = ...
    normalized: bool = ...
    def __getstate__(self): ...

@dataclass
class EncodingFast: ...

TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]

class TruncationStrategy(ExplicitEnum):
    ONLY_FIRST = ...
    ONLY_SECOND = ...
    LONGEST_FIRST = ...
    DO_NOT_TRUNCATE = ...

class CharSpan(NamedTuple):
    start: int
    end: int

...

class TokenSpan(NamedTuple):
    start: int
    end: int

...

class BatchEncoding(UserDict[str, Any]):
    def __init__(
        self,
        data: Optional[Dict[str, Any]] = ...,
        encoding: Optional[Union[EncodingFast, Sequence[EncodingFast]]] = ...,
        tensor_type: Union[None, str, TensorType] = ...,
        prepend_batch_axis: bool = ...,
        n_sequences: Optional[int] = ...,
    ) -> None: ...
    @property
    def n_sequences(self) -> Optional[int]: ...
    @property
    def is_fast(self) -> bool: ...
    def __getitem__(self, item: Union[int, str]) -> Union[Any, EncodingFast]: ...
    def __getattr__(self, item: str): ...
    def __getstate__(self): ...
    def __setstate__(self, state): ...
    def keys(self): ...
    def values(self): ...
    def items(self): ...
    @property
    def encodings(self) -> Optional[List[EncodingFast]]: ...
    def tokens(self, batch_index: int = ...) -> List[str]: ...
    def sequence_ids(self, batch_index: int = ...) -> List[Optional[int]]: ...
    def words(self, batch_index: int = ...) -> List[Optional[int]]: ...
    def word_ids(self, batch_index: int = ...) -> List[Optional[int]]: ...
    def token_to_sequence(
        self, batch_or_token_index: int, token_index: Optional[int] = ...
    ) -> int: ...
    def token_to_word(
        self, batch_or_token_index: int, token_index: Optional[int] = ...
    ) -> int: ...
    def word_to_tokens(
        self,
        batch_or_word_index: int,
        word_index: Optional[int] = ...,
        sequence_index: int = ...,
    ) -> Optional[TokenSpan]: ...
    def token_to_chars(
        self, batch_or_token_index: int, token_index: Optional[int] = ...
    ) -> CharSpan: ...
    def char_to_token(
        self,
        batch_or_char_index: int,
        char_index: Optional[int] = ...,
        sequence_index: int = ...,
    ) -> int: ...
    def word_to_chars(
        self,
        batch_or_word_index: int,
        word_index: Optional[int] = ...,
        sequence_index: int = ...,
    ) -> CharSpan: ...
    def char_to_word(
        self,
        batch_or_char_index: int,
        char_index: Optional[int] = ...,
        sequence_index: int = ...,
    ) -> int: ...
    def convert_to_tensors(
        self,
        tensor_type: Optional[Union[str, TensorType]] = ...,
        prepend_batch_axis: bool = ...,
    ): ...
    @torch_required
    def to(self, device: Union[str, torch.device]) -> BatchEncoding: ...

class SpecialTokensMixin:
    SPECIAL_TOKENS_ATTRIBUTES = ...
    def __init__(self, verbose=..., **kwargs) -> None: ...
    def sanitize_special_tokens(self) -> int: ...
    def add_special_tokens(
        self, special_tokens_dict: Dict[str, Union[str, AddedToken]]
    ) -> int: ...
    def add_tokens(
        self,
        new_tokens: Union[str, AddedToken, List[Union[str, AddedToken]]],
        special_tokens: bool = ...,
    ) -> int: ...
    @property
    def bos_token(self) -> str: ...
    @property
    def eos_token(self) -> str: ...
    @property
    def unk_token(self) -> str: ...
    @property
    def sep_token(self) -> str: ...
    @property
    def pad_token(self) -> str: ...
    @property
    def cls_token(self) -> str: ...
    @property
    def mask_token(self) -> str: ...
    @property
    def additional_special_tokens(self) -> List[str]: ...
    @bos_token.setter
    def bos_token(self, value): ...
    @eos_token.setter
    def eos_token(self, value): ...
    @unk_token.setter
    def unk_token(self, value): ...
    @sep_token.setter
    def sep_token(self, value): ...
    @pad_token.setter
    def pad_token(self, value): ...
    @cls_token.setter
    def cls_token(self, value): ...
    @mask_token.setter
    def mask_token(self, value): ...
    @additional_special_tokens.setter
    def additional_special_tokens(self, value): ...
    @property
    def bos_token_id(self) -> Optional[int]: ...
    @property
    def eos_token_id(self) -> Optional[int]: ...
    @property
    def unk_token_id(self) -> Optional[int]: ...
    @property
    def sep_token_id(self) -> Optional[int]: ...
    @property
    def pad_token_id(self) -> Optional[int]: ...
    @property
    def pad_token_type_id(self) -> int: ...
    @property
    def cls_token_id(self) -> Optional[int]: ...
    @property
    def mask_token_id(self) -> Optional[int]: ...
    @property
    def additional_special_tokens_ids(self) -> List[int]: ...
    @bos_token_id.setter
    def bos_token_id(self, value): ...
    @eos_token_id.setter
    def eos_token_id(self, value): ...
    @unk_token_id.setter
    def unk_token_id(self, value): ...
    @sep_token_id.setter
    def sep_token_id(self, value): ...
    @pad_token_id.setter
    def pad_token_id(self, value): ...
    @cls_token_id.setter
    def cls_token_id(self, value): ...
    @mask_token_id.setter
    def mask_token_id(self, value): ...
    @additional_special_tokens_ids.setter
    def additional_special_tokens_ids(self, values): ...
    @property
    def special_tokens_map(self) -> Dict[str, Union[str, List[str]]]: ...
    @property
    def special_tokens_map_extended(
        self,
    ) -> Dict[str, Union[str, AddedToken, List[Union[str, AddedToken]]]]: ...
    @property
    def all_special_tokens(self) -> List[str]: ...
    @property
    def all_special_tokens_extended(self) -> List[Union[str, AddedToken]]: ...
    @property
    def all_special_ids(self) -> List[int]: ...

ENCODE_KWARGS_DOCSTRING: str = ...
ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING: str = ...
INIT_TOKENIZER_DOCSTRING: str = ...

@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class PreTrainedTokenizerBase(SpecialTokensMixin, PushToHubMixin):
    vocab_files_names: Dict[str, str] = ...
    pretrained_vocab_files_map: Dict[str, Dict[str, str]] = ...
    pretrained_init_configuration: Dict[str, Dict[str, Any]] = ...
    max_model_input_sizes: Dict[str, Optional[int]] = ...
    model_input_names: List[str] = ...
    padding_side: str = ...
    slow_tokenizer_class = ...
    def __init__(self, **kwargs) -> None: ...
    @property
    def max_len_single_sentence(self) -> int: ...
    @property
    def max_len_sentences_pair(self) -> int: ...
    @max_len_single_sentence.setter
    def max_len_single_sentence(self, value) -> int: ...
    @max_len_sentences_pair.setter
    def max_len_sentences_pair(self, value) -> int: ...
    def __repr__(self) -> str: ...
    def get_vocab(self) -> Dict[str, int]: ...
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike[str]],
        *init_inputs: Any,
        **kwargs: Any
    ) -> "PreTrainedTokenizerBase": ...
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike[str]],
        legacy_format: Optional[bool] = ...,
        filename_prefix: Optional[str] = ...,
        push_to_hub: bool = ...,
        **kwargs: Any
    ) -> Tuple[str]: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = ...
    ) -> Tuple[str]: ...
    def tokenize(
        self,
        text: str,
        pair: Optional[str] = ...,
        add_special_tokens: bool = ...,
        **kwargs
    ) -> List[str]: ...
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING,
        """
            **kwargs: Passed along to the `.tokenize()` method.
        """,
        """
        Returns:
            :obj:`List[int]`, :obj:`torch.Tensor`, :obj:`tf.Tensor` or :obj:`np.ndarray`: The tokenized ids of the
            text.
        """,
    )
    def encode(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = ...,
        add_special_tokens: bool = ...,
        padding: Union[bool, str, PaddingStrategy] = ...,
        truncation: Union[bool, str, TruncationStrategy] = ...,
        max_length: Optional[int] = ...,
        stride: int = ...,
        return_tensors: Optional[Union[str, TensorType]] = ...,
        **kwargs
    ) -> List[int]: ...
    def num_special_tokens_to_add(self, pair: bool = ...) -> int: ...
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ],
        text_pair: Optional[
            Union[
                TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
            ]
        ] = ...,
        add_special_tokens: bool = ...,
        padding: Union[bool, str, PaddingStrategy] = ...,
        truncation: Union[bool, str, TruncationStrategy] = ...,
        max_length: Optional[int] = ...,
        stride: int = ...,
        is_split_into_words: bool = ...,
        pad_to_multiple_of: Optional[int] = ...,
        return_tensors: Optional[Union[str, TensorType]] = ...,
        return_token_type_ids: Optional[bool] = ...,
        return_attention_mask: Optional[bool] = ...,
        return_overflowing_tokens: bool = ...,
        return_special_tokens_mask: bool = ...,
        return_offsets_mapping: bool = ...,
        return_length: bool = ...,
        verbose: bool = ...,
        **kwargs
    ) -> BatchEncoding: ...
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = ...,
        add_special_tokens: bool = ...,
        padding: Union[bool, str, PaddingStrategy] = ...,
        truncation: Union[bool, str, TruncationStrategy] = ...,
        max_length: Optional[int] = ...,
        stride: int = ...,
        is_split_into_words: bool = ...,
        pad_to_multiple_of: Optional[int] = ...,
        return_tensors: Optional[Union[str, TensorType]] = ...,
        return_token_type_ids: Optional[bool] = ...,
        return_attention_mask: Optional[bool] = ...,
        return_overflowing_tokens: bool = ...,
        return_special_tokens_mask: bool = ...,
        return_offsets_mapping: bool = ...,
        return_length: bool = ...,
        verbose: bool = ...,
        **kwargs
    ) -> BatchEncoding: ...
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = ...,
        padding: Union[bool, str, PaddingStrategy] = ...,
        truncation: Union[bool, str, TruncationStrategy] = ...,
        max_length: Optional[int] = ...,
        stride: int = ...,
        is_split_into_words: bool = ...,
        pad_to_multiple_of: Optional[int] = ...,
        return_tensors: Optional[Union[str, TensorType]] = ...,
        return_token_type_ids: Optional[bool] = ...,
        return_attention_mask: Optional[bool] = ...,
        return_overflowing_tokens: bool = ...,
        return_special_tokens_mask: bool = ...,
        return_offsets_mapping: bool = ...,
        return_length: bool = ...,
        verbose: bool = ...,
        **kwargs
    ) -> BatchEncoding: ...
    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = ...,
        max_length: Optional[int] = ...,
        pad_to_multiple_of: Optional[int] = ...,
        return_attention_mask: Optional[bool] = ...,
        return_tensors: Optional[Union[str, TensorType]] = ...,
        verbose: bool = ...,
    ) -> BatchEncoding: ...
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...
    ) -> List[int]: ...
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...
    ) -> List[int]: ...
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = ...,
        add_special_tokens: bool = ...,
        padding: Union[bool, str, PaddingStrategy] = ...,
        truncation: Union[bool, str, TruncationStrategy] = ...,
        max_length: Optional[int] = ...,
        stride: int = ...,
        pad_to_multiple_of: Optional[int] = ...,
        return_tensors: Optional[Union[str, TensorType]] = ...,
        return_token_type_ids: Optional[bool] = ...,
        return_attention_mask: Optional[bool] = ...,
        return_overflowing_tokens: bool = ...,
        return_special_tokens_mask: bool = ...,
        return_offsets_mapping: bool = ...,
        return_length: bool = ...,
        verbose: bool = ...,
        prepend_batch_axis: bool = ...,
        **kwargs
    ) -> BatchEncoding: ...
    def truncate_sequences(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = ...,
        num_tokens_to_remove: int = ...,
        truncation_strategy: Union[str, TruncationStrategy] = ...,
        stride: int = ...,
    ) -> Tuple[List[int], List[int], List[int]]: ...
    def convert_tokens_to_string(self, tokens: List[str]) -> str: ...
    def batch_decode(
        self,
        sequences: Union[
            List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor
        ],
        skip_special_tokens: bool = ...,
        clean_up_tokenization_spaces: bool = ...,
        **kwargs
    ) -> List[str]: ...
    def decode(
        self,
        token_ids: Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor],
        skip_special_tokens: bool = ...,
        clean_up_tokenization_spaces: bool = ...,
        **kwargs
    ) -> str: ...
    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = ...,
        already_has_special_tokens: bool = ...,
    ) -> List[int]: ...
    @staticmethod
    def clean_up_tokenization(out_string: str) -> str: ...
    @contextmanager
    def as_target_tokenizer(self): ...
    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = ...,
        max_length: Optional[int] = ...,
        max_target_length: Optional[int] = ...,
        padding: str = ...,
        return_tensors: str = ...,
        truncation: bool = ...,
        **kwargs
    ) -> BatchEncoding: ...

def get_fast_tokenizer_file(
    path_or_repo: Union[str, os.PathLike],
    revision: Optional[str] = ...,
    use_auth_token: Optional[Union[bool, str]] = ...,
) -> str: ...
