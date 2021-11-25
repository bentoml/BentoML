

from typing import Dict, List, Optional

from ...tokenization_utils import PreTrainedTokenizer

logger = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
UNICODE_VOCAB_SIZE = ...
PAD = ...
CLS = ...
SEP = ...
BOS = ...
MASK = ...
RESERVED = ...
SPECIAL_CODEPOINTS: Dict[int, str] = ...
SPECIAL_CODEPOINTS_BY_NAME: Dict[str, int] = ...
class CanineTokenizer(PreTrainedTokenizer):
    r"""
    Construct a CANINE tokenizer (i.e. a character splitter). It turns text into a sequence of characters, and then
    converts each character into its Unicode code point.

    :class:`~transformers.CanineTokenizer` inherits from :class:`~transformers.PreTrainedTokenizer`.

    Refer to superclass :class:`~transformers.PreTrainedTokenizer` for usage examples and documentation concerning
    parameters.

    Args:
        model_max_length (:obj:`int`, `optional`, defaults to 2048):
                The maximum sentence length the model accepts.
    """
    max_model_input_sizes = ...
    def __init__(self, bos_token=..., eos_token=..., sep_token=..., cls_token=..., pad_token=..., mask_token=..., add_prefix_space=..., model_max_length=..., **kwargs) -> None:
        ...
    
    @property
    def vocab_size(self) -> int:
        ...
    
    def convert_tokens_to_string(self, tokens):
        ...
    
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A CANINE sequence has the following format:

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
    
    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ..., already_has_special_tokens: bool = ...) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        ...
    
    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A CANINE
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
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...):
        ...
    


