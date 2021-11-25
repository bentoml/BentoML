

from typing import List, Optional, Tuple

from ...tokenization_utils_fast import PreTrainedTokenizerFast

logger = ...
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
PRETRAINED_INIT_CONFIGURATION = ...
class MPNetTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" MPNet tokenizer (backed by HuggingFace's `tokenizers` library). Based on WordPiece.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizerFast` which contains most of the main
    methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        bos_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

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
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see `this
            issue <https://github.com/huggingface/transformers/issues/328>`__).
        strip_accents: (:obj:`bool`, `optional`):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for :obj:`lowercase` (as in the original BERT).
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    pretrained_init_configuration = ...
    max_model_input_sizes = ...
    slow_tokenizer_class = ...
    model_input_names = ...
    def __init__(self, vocab_file=..., tokenizer_file=..., do_lower_case=..., bos_token=..., eos_token=..., sep_token=..., cls_token=..., unk_token=..., pad_token=..., mask_token=..., tokenize_chinese_chars=..., strip_accents=..., **kwargs) -> None:
        ...
    
    @property
    def mask_token(self) -> str:
        """
        :obj:`str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while
        not having been set.

        MPNet tokenizer has a special mask token to be usable in the fill-mask pipeline. The mask token will greedily
        comprise the space before the `<mask>`.
        """
        ...
    
    @mask_token.setter
    def mask_token(self, value):
        """
        Overriding the default behavior of the mask token to have it eat the space before it.

        This is needed to preserve backward compatibility with all the previously used models based on MPNet.
        """
        ...
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=...):
        ...
    
    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. MPNet does not
        make use of token type ids, therefore a list of zeros is returned

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs

        Returns:
            :obj:`List[int]`: List of zeros.
        """
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    


