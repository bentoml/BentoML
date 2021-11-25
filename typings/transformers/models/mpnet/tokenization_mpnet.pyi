

from typing import List, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer

logger = ...
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
PRETRAINED_INIT_CONFIGURATION = ...
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    ...

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    ...

class MPNetTokenizer(PreTrainedTokenizer):
    """

    This tokenizer inherits from :class:`~transformers.BertTokenizer` which contains most of the methods. Users should
    refer to the superclass for more information regarding methods.

    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (:obj:`Iterable`, `optional`):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            :obj:`do_basic_tokenize=True`
        bos_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            The beginning of sequence token that was used during pre-training. Can be used a sequence classifier token.

            .. note::

                When building a sequence using special tokens, this is not the token that is used for the beginning of
                sequence. The token used is the :obj:`cls_token`.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sequence token.

            .. note::

                When building a sequence using special tokens, this is not the token that is used for the end of
                sequence. The token used is the :obj:`sep_token`.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this `issue
            <https://github.com/huggingface/transformers/issues/328>`__).
        strip_accents: (:obj:`bool`, `optional`):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for :obj:`lowercase` (as in the original BERT).
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    pretrained_init_configuration = ...
    max_model_input_sizes = ...
    model_input_names = ...
    def __init__(self, vocab_file, do_lower_case=..., do_basic_tokenize=..., never_split=..., bos_token=..., eos_token=..., sep_token=..., cls_token=..., unk_token=..., pad_token=..., mask_token=..., tokenize_chinese_chars=..., strip_accents=..., **kwargs) -> None:
        ...
    
    @property
    def do_lower_case(self):
        ...
    
    @property
    def vocab_size(self):
        ...
    
    def get_vocab(self):
        ...
    
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        ...
    
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A MPNet sequence has the following format:

        - single sequence: ``<s> X </s>``
        - pair of sequences: ``<s> A </s></s> B </s>``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        ...
    
    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ..., already_has_special_tokens: bool = ...) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` methods.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True if the token list is already formatted with special tokens for the model

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        ...
    
    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. MPNet does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of zeros.
        """
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    


class BasicTokenizer:
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        never_split (:obj:`Iterable`, `optional`):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            :obj:`do_basic_tokenize=True`
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this `issue
            <https://github.com/huggingface/transformers/issues/328>`__).
        strip_accents: (:obj:`bool`, `optional`):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for :obj:`lowercase` (as in the original BERT).
    """
    def __init__(self, do_lower_case=..., never_split=..., tokenize_chinese_chars=..., strip_accents=...) -> None:
        ...
    
    def tokenize(self, text, never_split=...):
        """
        Basic Tokenization of a piece of text. Split on "white spaces" only, for sub-word tokenization, see
        WordPieceTokenizer.

        Args:
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                :func:`PreTrainedTokenizer.tokenize`) List of token not to split.
        """
        ...
    


class WordpieceTokenizer:
    """Runs WordPiece tokenization."""
    def __init__(self, vocab, unk_token, max_input_chars_per_word=...) -> None:
        ...
    
    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, :obj:`input = "unaffable"` wil return as output :obj:`["un", "##aff", "##able"]`.

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """
        ...
    


