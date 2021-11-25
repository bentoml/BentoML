

from typing import Dict, List, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer

logger = ...
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
PRETRAINED_INIT_CONFIGURATION = ...
def get_pairs(word):
    """
    Return set of symbol pairs in a word. word is represented as tuple of symbols (symbols being variable-length
    strings)
    """
    ...

def replace_unicode_punct(text):
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/replace-unicode-punctuation.perl
    """
    ...

def remove_non_printing_char(text):
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/remove-non-printing-char.perl
    """
    ...

class FSMTTokenizer(PreTrainedTokenizer):
    """
    Construct an FAIRSEQ Transformer tokenizer. Based on Byte-Pair Encoding. The tokenization process is the following:

    - Moses preprocessing and tokenization.
    - Normalizing all inputs text.
    - The arguments ``special_tokens`` and the function ``set_special_tokens``, can be used to add additional symbols
      (like "__classify__") to a vocabulary.
    - The argument :obj:`langs` defines a pair of languages.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        langs (:obj:`List[str]`):
            A list of two languages to translate from and to, for instance :obj:`["en", "ru"]`.
        src_vocab_file (:obj:`str`):
            File containing the vocabulary for the source language.
        tgt_vocab_file (:obj:`st`):
            File containing the vocabulary for the target language.
        merges_file (:obj:`str`):
            File containing the merges.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            .. note::

                When building a sequence using special tokens, this is not the token that is used for the beginning of
                sequence. The token used is the :obj:`cls_token`.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.

    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    pretrained_init_configuration = ...
    max_model_input_sizes = ...
    model_input_names = ...
    def __init__(self, langs=..., src_vocab_file=..., tgt_vocab_file=..., merges_file=..., do_lower_case=..., unk_token=..., bos_token=..., sep_token=..., pad_token=..., **kwargs) -> None:
        ...
    
    def get_vocab(self) -> Dict[str, int]:
        ...
    
    @property
    def vocab_size(self) -> int:
        ...
    
    def moses_punct_norm(self, text, lang):
        ...
    
    def moses_tokenize(self, text, lang):
        ...
    
    def moses_detokenize(self, tokens, lang):
        ...
    
    def moses_pipeline(self, text, lang):
        ...
    
    @property
    def src_vocab_size(self):
        ...
    
    @property
    def tgt_vocab_size(self):
        ...
    
    def get_src_vocab(self):
        ...
    
    def get_tgt_vocab(self):
        ...
    
    def bpe(self, token):
        ...
    
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        ...
    
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A FAIRSEQ Transformer sequence has the following format:

        - single sequence: ``<s> X </s>``
        - pair of sequences: ``<s> A </s> B </s>``

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
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A FAIRSEQ
        Transformer sequence pair mask has the following format:

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

        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An
        FAIRSEQ_TRANSFORMER sequence pair mask has the following format:
        """
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    


