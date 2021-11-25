

from typing import List, Optional

from ..bert.tokenization_bert import BertTokenizer

logger = ...
VOCAB_FILES_NAMES = ...
_model_names = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
PRETRAINED_INIT_CONFIGURATION = ...
class FunnelTokenizer(BertTokenizer):
    r"""
    Construct a Funnel Transformer tokenizer.

    :class:`~transformers.FunnelTokenizer` is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting and wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    pretrained_init_configuration = ...
    cls_token_type_id: int = ...
    def __init__(self, vocab_file, do_lower_case=..., do_basic_tokenize=..., never_split=..., unk_token=..., sep_token=..., pad_token=..., cls_token=..., mask_token=..., bos_token=..., eos_token=..., tokenize_chinese_chars=..., strip_accents=..., **kwargs) -> None:
        ...
    
    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A Funnel
        Transformer sequence pair mask has the following format:

        ::

            2 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
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
    


