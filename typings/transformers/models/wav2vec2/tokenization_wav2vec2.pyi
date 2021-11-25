

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...file_utils import PaddingStrategy, TensorType, add_end_docstrings
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding

logger = ...
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
WAV2VEC2_KWARGS_DOCSTRING = ...
class Wav2Vec2CTCTokenizer(PreTrainedTokenizer):
    """
    Constructs a Wav2Vec2CTC tokenizer.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains some of the main methods.
    Users should refer to the superclass for more information regarding such methods.

    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        bos_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            The beginning of sentence token.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sentence token.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        word_delimiter_token (:obj:`str`, `optional`, defaults to :obj:`"|"`):
            The token used for defining the end of a word.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to accept lowercase input and lowercase the output when decoding.

        **kwargs
            Additional keyword arguments passed along to :class:`~transformers.PreTrainedTokenizer`
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    model_input_names = ...
    def __init__(self, vocab_file, bos_token=..., eos_token=..., unk_token=..., pad_token=..., word_delimiter_token=..., do_lower_case=..., **kwargs) -> None:
        ...
    
    @property
    def word_delimiter_token(self) -> str:
        """
        :obj:`str`: Word delimiter token. Log an error if used while not having been set.
        """
        ...
    
    @property
    def word_delimiter_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the word_delimiter_token in the vocabulary. Returns :obj:`None` if the token has
        not been set.
        """
        ...
    
    @word_delimiter_token.setter
    def word_delimiter_token(self, value):
        ...
    
    @word_delimiter_token_id.setter
    def word_delimiter_token_id(self, value):
        ...
    
    @property
    def vocab_size(self) -> int:
        ...
    
    def get_vocab(self) -> Dict:
        ...
    
    def convert_tokens_to_string(self, tokens: List[str], group_tokens: bool = ..., spaces_between_special_tokens: bool = ...) -> str:
        """
        Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
        """
        ...
    
    def prepare_for_tokenization(self, text, is_split_into_words=..., **kwargs):
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    


class Wav2Vec2Tokenizer(PreTrainedTokenizer):
    """
    Constructs a Wav2Vec2 tokenizer.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains some of the main methods.
    Users should refer to the superclass for more information regarding such methods.

    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        bos_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            The beginning of sentence token.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sentence token.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        word_delimiter_token (:obj:`str`, `optional`, defaults to :obj:`"|"`):
            The token used for defining the end of a word.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to lowercase the output when decoding.
        do_normalize (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly
            improve the performance for some models, *e.g.*, `wav2vec2-lv60
            <https://huggingface.co/models?search=lv60>`__.
        return_attention_mask (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not :meth:`~transformers.Wav2Vec2Tokenizer.__call__` should return :obj:`attention_mask`.

            .. note::

                Wav2Vec2 models that have set ``config.feat_extract_norm == "group"``, such as `wav2vec2-base
                <https://huggingface.co/facebook/wav2vec2-base-960h>`__, have **not** been trained using
                :obj:`attention_mask`. For such models, :obj:`input_values` should simply be padded with 0 and no
                :obj:`attention_mask` should be passed.

                For Wav2Vec2 models that have set ``config.feat_extract_norm == "layer"``, such as `wav2vec2-lv60
                <https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self>`__, :obj:`attention_mask` should be
                passed for batched inference.

        **kwargs
            Additional keyword arguments passed along to :class:`~transformers.PreTrainedTokenizer`
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    model_input_names = ...
    def __init__(self, vocab_file, bos_token=..., eos_token=..., unk_token=..., pad_token=..., word_delimiter_token=..., do_lower_case=..., do_normalize=..., return_attention_mask=..., **kwargs) -> None:
        ...
    
    @property
    def word_delimiter_token(self) -> str:
        """
        :obj:`str`: Padding token. Log an error if used while not having been set.
        """
        ...
    
    @property
    def word_delimiter_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the word_delimiter_token in the vocabulary. Returns :obj:`None` if the token has
        not been set.
        """
        ...
    
    @word_delimiter_token.setter
    def word_delimiter_token(self, value):
        ...
    
    @word_delimiter_token_id.setter
    def word_delimiter_token_id(self, value):
        ...
    
    @add_end_docstrings(WAV2VEC2_KWARGS_DOCSTRING)
    def __call__(self, raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]], padding: Union[bool, str, PaddingStrategy] = ..., max_length: Optional[int] = ..., pad_to_multiple_of: Optional[int] = ..., return_tensors: Optional[Union[str, TensorType]] = ..., verbose: bool = ..., **kwargs) -> BatchEncoding:
        """
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences.

        Args:
            raw_speech (:obj:`np.ndarray`, :obj:`List[float]`, :obj:`List[np.ndarray]`, :obj:`List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrayr or a list of list of float values.
        """
        ...
    
    @property
    def vocab_size(self) -> int:
        ...
    
    def get_vocab(self) -> Dict:
        ...
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
        """
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    


