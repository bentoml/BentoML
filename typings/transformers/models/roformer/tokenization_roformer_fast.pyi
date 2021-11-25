

from typing import List, Optional, Tuple

from ...tokenization_utils_fast import PreTrainedTokenizerFast

logger = ...
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
PRETRAINED_INIT_CONFIGURATION = ...
class RoFormerTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" RoFormer tokenizer (backed by HuggingFace's `tokenizers` library).

    :class:`~transformers.RoFormerTokenizerFast` is almost identical to :class:`~transformers.BertTokenizerFast` and
    runs end-to-end tokenization: punctuation splitting and wordpiece. There are some difference between them when
    tokenizing Chinese.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizerFast` which contains most of the main
    methods. Users should refer to this superclass for more information regarding those methods.

    Example::

        >>> from transformers import RoFormerTokenizerFast
        >>> tokenizer = RoFormerTokenizerFast.from_pretrained('junnyu/roformer_chinese_base')
        >>> tokenizer.tokenize("今天天气非常好。")
        # ['今', '天', '天', '气', '非常', '好', '。']

    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    pretrained_init_configuration = ...
    slow_tokenizer_class = ...
    def __init__(self, vocab_file=..., tokenizer_file=..., do_lower_case=..., unk_token=..., sep_token=..., pad_token=..., cls_token=..., mask_token=..., tokenize_chinese_chars=..., strip_accents=..., **kwargs) -> None:
        ...
    
    def __getstate__(self):
        ...
    
    def __setstate__(self, d):
        ...
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=...):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A RoFormer sequence has the following format:

        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        ...
    
    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A RoFormer
        sequence pair mask has the following format:

        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    
    def save_pretrained(self, save_directory, legacy_format=..., filename_prefix=..., push_to_hub=..., **kwargs):
        ...
    


