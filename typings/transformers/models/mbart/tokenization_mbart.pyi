

from contextlib import contextmanager
from typing import List, Optional

from ...tokenization_utils import BatchEncoding
from ..xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer

logger = ...
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
FAIRSEQ_LANGUAGE_CODES = ...
class MBartTokenizer(XLMRobertaTokenizer):
    """
    Construct an MBART tokenizer.

    :class:`~transformers.MBartTokenizer` is a subclass of :class:`~transformers.XLMRobertaTokenizer`. Refer to
    superclass :class:`~transformers.XLMRobertaTokenizer` for usage examples and documentation concerning the
    initialization parameters and other methods.

    The tokenization method is ``<tokens> <eos> <language code>`` for source language documents, and ``<language code>
    <tokens> <eos>``` for target language documents.

    Examples::

        >>> from transformers import MBartTokenizer
        >>> tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-en-ro', src_lang="en_XX", tgt_lang="ro_RO")
        >>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
        >>> expected_translation_romanian = "Şeful ONU declară că nu există o soluţie militară în Siria"
        >>> inputs = tokenizer(example_english_phrase, return_tensors="pt)
        >>> with tokenizer.as_target_tokenizer():
        ...     labels = tokenizer(expected_translation_romanian, return_tensors="pt")
        >>> inputs["labels"] = labels["input_ids"]
    """
    vocab_files_names = ...
    max_model_input_sizes = ...
    pretrained_vocab_files_map = ...
    prefix_tokens: List[int] = ...
    suffix_tokens: List[int] = ...
    def __init__(self, *args, tokenizer_file=..., src_lang=..., tgt_lang=..., additional_special_tokens=..., **kwargs) -> None:
        ...
    
    @property
    def vocab_size(self):
        ...
    
    @property
    def src_lang(self) -> str:
        ...
    
    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
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
        adding special tokens. An MBART sequence has the following format, where ``X`` represents the sequence:

        - ``input_ids`` (for encoder) ``X [eos, src_lang_code]``
        - ``decoder_input_ids``: (for decoder) ``X [eos, tgt_lang_code]``

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
    
    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting. No prefix and suffix=[eos, src_lang_code]."""
        ...
    
    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target language setting. No prefix and suffix=[eos, tgt_lang_code]."""
        ...
    


