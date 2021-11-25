

from typing import Any, Dict, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer

logger = ...
SPIECE_UNDERLINE = ...
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
class ReformerTokenizer(PreTrainedTokenizer):
    """
    Construct a Reformer tokenizer. Based on `SentencePiece <https://github.com/google/sentencepiece>`__ .

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a `.spm` extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sequence token.

            .. note::

                When building a sequence using special tokens, this is not the token that is used for the end of
                sequence. The token used is the :obj:`sep_token`.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        additional_special_tokens (:obj:`List[str]`, `optional`):
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
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    model_input_names = ...
    def __init__(self, vocab_file, eos_token=..., unk_token=..., additional_special_tokens=..., sp_model_kwargs: Optional[Dict[str, Any]] = ..., **kwargs) -> None:
        ...
    
    @property
    def vocab_size(self):
        ...
    
    def get_vocab(self) -> Dict[str, int]:
        ...
    
    def __getstate__(self):
        ...
    
    def __setstate__(self, d):
        ...
    
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    


