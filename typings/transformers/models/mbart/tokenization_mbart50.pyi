

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

from ...tokenization_utils import BatchEncoding, PreTrainedTokenizer

logger = ...
SPIECE_UNDERLINE = ...
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
FAIRSEQ_LANGUAGE_CODES = ...
class MBart50Tokenizer(PreTrainedTokenizer):
    """
    Construct a MBart50 tokenizer. Based on `SentencePiece <https://github.com/google/sentencepiece>`__.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        src_lang (:obj:`str`, `optional`):
            A string representing the source language.
        tgt_lang (:obj:`str`, `optional`):
            A string representing the target language.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sequence token.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"<mask>"`):
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

    Examples::

        >>> from transformers import MBart50Tokenizer
        >>> tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ro_RO")
        >>> src_text = " UN Chief Says There Is No Military Solution in Syria"
        >>> tgt_text =  "Şeful ONU declară că nu există o soluţie militară în Siria"
        >>> model_inputs = tokenizer(src_text, return_tensors="pt")
        >>> with tokenizer.as_target_tokenizer():
        ...    labels = tokenizer(tgt_text, return_tensors="pt").input_ids
        >>> # model(**model_inputs, labels=labels) should work
    """
    vocab_files_names = ...
    max_model_input_sizes = ...
    pretrained_vocab_files_map = ...
    model_input_names = ...
    prefix_tokens: List[int] = ...
    suffix_tokens: List[int] = ...
    def __init__(self, vocab_file, src_lang=..., tgt_lang=..., eos_token=..., sep_token=..., cls_token=..., unk_token=..., pad_token=..., mask_token=..., sp_model_kwargs: Optional[Dict[str, Any]] = ..., **kwargs) -> None:
        ...
    
    @property
    def vocab_size(self) -> int:
        ...
    
    @property
    def src_lang(self) -> str:
        ...
    
    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        ...
    
    def __getstate__(self) -> Dict:
        ...
    
    def __setstate__(self, d: Dict) -> None:
        ...
    
    def get_vocab(self) -> Dict:
        ...
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
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
    
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An MBART-50 sequence has the following format, where ``X`` represents the sequence:

        - ``input_ids`` (for encoder) ``[src_lang_code] X [eos]``
        - ``labels``: (for decoder) ``[tgt_lang_code] X [eos]``

        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
        separator.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        ...
    
    def prepare_seq2seq_batch(self, src_texts: List[str], src_lang: str = ..., tgt_texts: Optional[List[str]] = ..., tgt_lang: str = ..., **kwargs) -> BatchEncoding:
        ...
    
    @contextmanager
    def as_target_tokenizer(self):
        """
        Temporarily sets the tokenizer for encoding the targets. Useful for tokenizer associated to
        sequence-to-sequence models that need a slightly different processing for the labels.
        """
        ...
    
    def set_src_lang_special_tokens(self, src_lang: str) -> None:
        """Reset the special tokens to the source lang setting. prefix=[src_lang_code] and suffix=[eos]."""
        ...
    
    def set_tgt_lang_special_tokens(self, tgt_lang: str) -> None:
        """Reset the special tokens to the target language setting. prefix=[tgt_lang_code] and suffix=[eos]."""
        ...
    


