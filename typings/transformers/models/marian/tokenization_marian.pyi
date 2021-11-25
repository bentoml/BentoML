

import re
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import sentencepiece

from ...tokenization_utils import PreTrainedTokenizer

VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
PRETRAINED_INIT_CONFIGURATION = ...
class MarianTokenizer(PreTrainedTokenizer):
    r"""
    Construct a Marian tokenizer. Based on `SentencePiece <https://github.com/google/sentencepiece>`__.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        source_spm (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a .spm extension) that
            contains the vocabulary for the source language.
        target_spm (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a .spm extension) that
            contains the vocabulary for the target language.
        source_lang (:obj:`str`, `optional`):
            A string representing the source language.
        target_lang (:obj:`str`, `optional`):
            A string representing the target language.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sequence token.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        model_max_length (:obj:`int`, `optional`, defaults to 512):
            The maximum sentence length the model accepts.
        additional_special_tokens (:obj:`List[str]`, `optional`, defaults to :obj:`["<eop>", "<eod>"]`):
            Additional special tokens used by the tokenizer.
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

        >>> from transformers import MarianTokenizer
        >>> tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
        >>> src_texts = [ "I am a small frog.", "Tom asked his teacher for advice."]
        >>> tgt_texts = ["Ich bin ein kleiner Frosch.", "Tom bat seinen Lehrer um Rat."]  # optional
        >>> inputs = tokenizer(src_texts, return_tensors="pt", padding=True)
        >>> with tokenizer.as_target_tokenizer():
        ...     labels = tokenizer(tgt_texts, return_tensors="pt", padding=True)
        >>> inputs["labels"] = labels["input_ids"]
        # keys  [input_ids, attention_mask, labels].
        >>> outputs = model(**inputs) should work
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    pretrained_init_configuration = ...
    max_model_input_sizes = ...
    model_input_names = ...
    language_code_re: re.Pattern = ...
    def __init__(self, vocab, source_spm, target_spm, source_lang=..., target_lang=..., unk_token=..., eos_token=..., pad_token=..., model_max_length=..., sp_model_kwargs: Optional[Dict[str, Any]] = ..., **kwargs) -> None:
        ...
    
    def normalize(self, x: str) -> str:
        """Cover moses empty string edge case. They return empty list for '' input!"""
        ...
    
    def remove_language_code(self, text: str):
        """Remove language codes like >>fr<< before sentencepiece"""
        ...
    
    def batch_decode(self, sequences, **kwargs):
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (:obj:`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the ``__call__`` method.
            skip_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to clean up the tokenization spaces.
            use_source_tokenizer (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use the source tokenizer to decode sequences (only applicable in sequence-to-sequence
                problems).
            kwargs (additional keyword arguments, `optional`):
                Will be passed to the underlying model specific decode method.

        Returns:
            :obj:`List[str]`: The list of decoded sentences.
        """
        ...
    
    def decode(self, token_ids, **kwargs):
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing ``self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))``.

        Args:
            token_ids (:obj:`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the ``__call__`` method.
            skip_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to clean up the tokenization spaces.
            use_source_tokenizer (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use the source tokenizer to decode sequences (only applicable in sequence-to-sequence
                problems).
            kwargs (additional keyword arguments, `optional`):
                Will be passed to the underlying model specific decode method.

        Returns:
            :obj:`str`: The decoded sentence.
        """
        ...
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Uses source spm if _decode_use_source_tokenizer is True, and target spm otherwise"""
        ...
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=...) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        ...
    
    @contextmanager
    def as_target_tokenizer(self):
        """
        Temporarily sets the tokenizer for encoding the targets. Useful for tokenizer associated to
        sequence-to-sequence models that need a slightly different processing for the labels.
        """
        ...
    
    @property
    def vocab_size(self) -> int:
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    
    def get_vocab(self) -> Dict:
        ...
    
    def __getstate__(self) -> Dict:
        ...
    
    def __setstate__(self, d: Dict) -> None:
        ...
    
    def num_special_tokens_to_add(self, **unused):
        """Just EOS"""
        ...
    
    def get_special_tokens_mask(self, token_ids_0: List, token_ids_1: Optional[List] = ..., already_has_special_tokens: bool = ...) -> List[int]:
        """Get list where entries are [1] if a token is [eos] or [pad] else 0."""
        ...
    


def load_spm(path: str, sp_model_kwargs: Dict[str, Any]) -> sentencepiece.SentencePieceProcessor:
    ...

def save_json(data, path: str) -> None:
    ...

def load_json(path: str) -> Union[Dict, List]:
    ...

