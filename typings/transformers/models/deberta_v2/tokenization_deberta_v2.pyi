

from typing import Any, Dict, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer

PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
PRETRAINED_INIT_CONFIGURATION = ...
VOCAB_FILES_NAMES = ...
class DebertaV2Tokenizer(PreTrainedTokenizer):
    r"""
    Constructs a DeBERTa-v2 tokenizer. Based on `SentencePiece <https://github.com/google/sentencepiece>`__.

    Args:
        vocab_file (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a `.spm` extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to lowercase the input when tokenizing.
        bos_token (:obj:`string`, `optional`, defaults to "[CLS]"):
            The beginning of sequence token that was used during pre-training. Can be used a sequence classifier token.
            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the :obj:`cls_token`.
        eos_token (:obj:`string`, `optional`, defaults to "[SEP]"):
            The end of sequence token. When building a sequence using special tokens, this is not the token that is
            used for the end of sequence. The token used is the :obj:`sep_token`.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        sp_model_kwargs (:obj:`dict`, `optional`):
            Will be passed to the ``SentencePieceProcessor.__init__()`` method. The `Python wrapper for SentencePiece
            <https://github.com/google/sentencepiece/tree/master/python>`__ can be used, among other things, to set:

            - ``enable_sampling``: Enable subword regularization.
            - ``nbest_size``: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - ``nbest_size = {0,1}``: No sampling is performed.
              - ``nbest_size > 1``: samples from the nbest_size results.
              - ``nbest_size < 0``: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - ``alpha``: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    pretrained_init_configuration = ...
    max_model_input_sizes = ...
    def __init__(self, vocab_file, do_lower_case=..., split_by_punct=..., bos_token=..., eos_token=..., unk_token=..., sep_token=..., pad_token=..., cls_token=..., mask_token=..., sp_model_kwargs: Optional[Dict[str, Any]] = ..., **kwargs) -> None:
        ...
    
    @property
    def vocab_size(self):
        ...
    
    @property
    def vocab(self):
        ...
    
    def get_vocab(self):
        ...
    
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        ...
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=...):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A DeBERTa sequence has the following format:

        - single sequence: [CLS] X [SEP]
        - pair of sequences: [CLS] A [SEP] B [SEP]

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        ...
    
    def get_special_tokens_mask(self, token_ids_0, token_ids_1=..., already_has_special_tokens=...):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

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
    
    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=...):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A DeBERTa
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
    
    def prepare_for_tokenization(self, text, is_split_into_words=..., **kwargs):
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    


class SPMTokenizer:
    r"""
    Constructs a tokenizer based on `SentencePiece <https://github.com/google/sentencepiece>`__.

    Args:
        vocab_file (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a `.spm` extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        sp_model_kwargs (:obj:`dict`, `optional`):
            Will be passed to the ``SentencePieceProcessor.__init__()`` method. The `Python wrapper for SentencePiece
            <https://github.com/google/sentencepiece/tree/master/python>`__ can be used, among other things, to set:

            - ``enable_sampling``: Enable subword regularization.
            - ``nbest_size``: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - ``nbest_size = {0,1}``: No sampling is performed.
              - ``nbest_size > 1``: samples from the nbest_size results.
              - ``nbest_size < 0``: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - ``alpha``: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
    """
    def __init__(self, vocab_file, split_by_punct=..., sp_model_kwargs: Optional[Dict[str, Any]] = ...) -> None:
        ...
    
    def __getstate__(self):
        ...
    
    def __setstate__(self, d):
        ...
    
    def tokenize(self, text):
        ...
    
    def convert_ids_to_tokens(self, ids):
        ...
    
    def decode(self, tokens, start=..., end=..., raw_text=...):
        ...
    
    def add_special_token(self, token):
        ...
    
    def part_of_whole_word(self, token, is_bos=...):
        ...
    
    def pad(self):
        ...
    
    def bos(self):
        ...
    
    def eos(self):
        ...
    
    def unk(self):
        ...
    
    def mask(self):
        ...
    
    def sym(self, id):
        ...
    
    def id(self, sym):
        ...
    
    def split_to_words(self, text):
        ...
    
    def save_pretrained(self, path: str, filename_prefix: str = ...):
        ...
    


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    ...

