

from typing import TYPE_CHECKING, List

from ..roberta.tokenization_roberta import RobertaTokenizer

if TYPE_CHECKING:
    ...
logger = ...
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
class BlenderbotTokenizer(RobertaTokenizer):
    r"""
    Construct a Blenderbot tokenizer.

    :class:`~transformers.Blenderbot` is nearly identical to :class:`~transformers.RobertaTokenizer` and runs
    end-to-end tokenization: punctuation splitting and wordpiece. The only difference is that it doesn't add BOS token
    to the beginning of sequences.

    Refer to superclass :class:`~transformers.RobertaTokenizer` for usage examples and documentation concerning
    parameters.
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: List[int] = ...):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A Blenderbot sequence has the following format:

        - single sequence: `` X </s>``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`):
                Will be ignored

        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        ...
    


def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    ...

