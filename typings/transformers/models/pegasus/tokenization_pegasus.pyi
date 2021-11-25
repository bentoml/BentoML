

from typing import Any, Dict, List, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer

SPIECE_UNDERLINE = ...
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
logger = ...
class PegasusTokenizer(PreTrainedTokenizer):
    r"""
    Construct a PEGASUS tokenizer. Based on `SentencePiece <https://github.com/google/sentencepiece>`__.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a `.spm` extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sequence token.

            .. note::

                When building a sequence using special tokens, this is not the token that is used for the end of
                sequence. The token used is the :obj:`sep_token`.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"<mask_2>"`):
            The token used for masking single token values. This is the token used when training this model with masked
            language modeling (MLM). This is the token that the PEGASUS encoder will try to predict during pretraining.
            It corresponds to `[MASK2]` in `PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive
            Summarization <https://arxiv.org/pdf/1912.08777.pdf>`__.
        mask_token_sent (:obj:`str`, `optional`, defaults to :obj:`"<mask_1>"`):
            The token used for masking whole target sentences. This is the token used when training this model with gap
            sentences generation (GSG). This is the sentence that the PEGASUS decoder will try to predict during
            pretraining. It corresponds to `[MASK1]` in `PEGASUS: Pre-training with Extracted Gap-sentences for
            Abstractive Summarization <https://arxiv.org/pdf/1912.08777.pdf>`__.
        additional_special_tokens (:obj:`List[str]`, `optional`):
            Additional special tokens used by the tokenizer. If no additional_special_tokens are provided <mask_2> and
            <unk_2, ..., unk_102> are used as additional special tokens corresponding to the `original PEGASUS
            tokenizer
            <https://github.com/google-research/pegasus/blob/939830367bcf411193d2b5eca2f2f90f3f9260ca/pegasus/ops/pretrain_parsing_ops.cc#L66>`__
            that uses the tokens 2 - 104 only for pretraining
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
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    model_input_names = ...
    def __init__(self, vocab_file, pad_token=..., eos_token=..., unk_token=..., mask_token=..., mask_token_sent=..., additional_special_tokens=..., offset=..., sp_model_kwargs: Optional[Dict[str, Any]] = ..., **kwargs) -> None:
        ...
    
    @property
    def vocab_size(self) -> int:
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
    
    def num_special_tokens_to_add(self, pair=...):
        """Just EOS"""
        ...
    
    def get_special_tokens_mask(self, token_ids_0: List, token_ids_1: Optional[List] = ..., already_has_special_tokens: bool = ...) -> List[int]:
        """Get list where entries are [1] if a token is [eos] or [pad] else 0."""
        ...
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=...) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating
        and adding special tokens. A PEGASUS sequence has the following format, where ``X`` represents the sequence:

        - single sequence: ``X </s>``
        - pair of sequences: ``A B </s>`` (not intended use)

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
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    


