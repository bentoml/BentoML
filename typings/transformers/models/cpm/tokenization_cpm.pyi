

from ..xlnet.tokenization_xlnet import XLNetTokenizer

logger = ...
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
class CpmTokenizer(XLNetTokenizer):
    """Runs pre-tokenization with Jieba segmentation tool. It is used in CPM models."""
    def __init__(self, *args, **kwargs) -> None:
        """
        Construct a CPM tokenizer. Based on `Jieba <https://pypi.org/project/jieba/>` and `SentencePiece
        <https://github.com/google/sentencepiece>`__.

        This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main
        methods. Users should refer to this superclass for more information regarding those methods.

        Args:
            vocab_file (:obj:`str`):
                `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a .spm extension) that
                contains the vocabulary necessary to instantiate a tokenizer.
            do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether to lowercase the input when tokenizing.
            remove_space (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether to strip the text when tokenizing (removing excess spaces before and after the string).
            keep_accents (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to keep accents when tokenizing.
            bos_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
                The beginning of sequence token that was used during pretraining. Can be used a sequence classifier
                token.

                .. note::

                    When building a sequence using special tokens, this is not the token that is used for the beginning
                    of sequence. The token used is the :obj:`cls_token`.
            eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
                The end of sequence token.

                .. note::

                    When building a sequence using special tokens, this is not the token that is used for the end of
                    sequence. The token used is the :obj:`sep_token`.
            unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
                The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be
                this token instead.
            sep_token (:obj:`str`, `optional`, defaults to :obj:`"<sep>"`):
                The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences
                for sequence classification or for a text and a question for question answering. It is also used as the
                last token of a sequence built with special tokens.
            pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
                The token used for padding, for example when batching sequences of different lengths.
            cls_token (:obj:`str`, `optional`, defaults to :obj:`"<cls>"`):
                The classifier token which is used when doing sequence classification (classification of the whole
                sequence instead of per-token classification). It is the first token of the sequence when built with
                special tokens.
            mask_token (:obj:`str`, `optional`, defaults to :obj:`"<mask>"`):
                The token used for masking values. This is the token used when training this model with masked language
                modeling. This is the token which the model will try to predict.
            additional_special_tokens (:obj:`List[str]`, `optional`, defaults to :obj:`["<eop>", "<eod>"]`):
                Additional special tokens used by the tokenizer.

        Attributes:
            sp_model (:obj:`SentencePieceProcessor`):
                The `SentencePiece` processor that is used for every conversion (string, tokens and IDs).
        """
        ...
    


