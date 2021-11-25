

from typing import List, Optional, Union

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...file_utils import PaddingStrategy, TensorType

logger = ...
class Wav2Vec2FeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Wav2Vec2 feature extractor.

    This feature extractor inherits from
    :class:`~transformers.feature_extraction_sequence_utils.SequenceFeatureExtractor` which contains most of the main
    methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (:obj:`int`, defaults to 1):
            The feature dimension of the extracted features.
        sampling_rate (:obj:`int`, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in Hertz per second (Hz).
        padding_value (:obj:`float`, defaults to 0.0):
            The value that is used to fill the padding values.
        do_normalize (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly
            improve the performance for some models, *e.g.*, `wav2vec2-lv60
            <https://huggingface.co/models?search=lv60>`__.
        return_attention_mask (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not :meth:`~transformers.Wav2Vec2FeatureExtractor.__call__` should return :obj:`attention_mask`.

            .. note::

                Wav2Vec2 models that have set ``config.feat_extract_norm == "group"``, such as `wav2vec2-base
                <https://huggingface.co/facebook/wav2vec2-base-960h>`__, have **not** been trained using
                :obj:`attention_mask`. For such models, :obj:`input_values` should simply be padded with 0 and no
                :obj:`attention_mask` should be passed.

                For Wav2Vec2 models that have set ``config.feat_extract_norm == "layer"``, such as `wav2vec2-lv60
                <https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self>`__, :obj:`attention_mask` should be
                passed for batched inference.
    """
    model_input_names = ...
    def __init__(self, feature_size=..., sampling_rate=..., padding_value=..., return_attention_mask=..., do_normalize=..., **kwargs) -> None:
        ...
    
    @staticmethod
    def zero_mean_unit_var_norm(input_values: List[np.ndarray]) -> List[np.ndarray]:
        """
        Every array in the list is normalized to have zero mean and unit variance
        """
        ...
    
    def __call__(self, raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]], padding: Union[bool, str, PaddingStrategy] = ..., max_length: Optional[int] = ..., pad_to_multiple_of: Optional[int] = ..., return_attention_mask: Optional[bool] = ..., return_tensors: Optional[Union[str, TensorType]] = ..., sampling_rate: Optional[int] = ..., **kwargs) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s). sequences.

        Args:
            raw_speech (:obj:`np.ndarray`, :obj:`List[float]`, :obj:`List[np.ndarray]`, :obj:`List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
                  single sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                >= 7.5 (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (:obj:`bool`, `optional`):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                `What are attention masks? <../glossary.html#attention-mask>`__

                .. note::

                    Wav2Vec2 models that have set ``config.feat_extract_norm == "group"``, such as `wav2vec2-base
                    <https://huggingface.co/facebook/wav2vec2-base-960h>`__, have **not** been trained using
                    :obj:`attention_mask`. For such models, :obj:`input_values` should simply be padded with 0 and no
                    :obj:`attention_mask` should be passed.

                    For Wav2Vec2 models that have set ``config.feat_extract_norm == "layer"``, such as `wav2vec2-lv60
                    <https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self>`__, :obj:`attention_mask` should be
                    passed for batched inference.

            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
            sampling_rate (:obj:`int`, `optional`):
                The sampling rate at which the ``raw_speech`` input was sampled. It is strongly recommended to pass
                ``sampling_rate`` at the forward call to prevent silent errors.
            padding_value (:obj:`float`, defaults to 0.0):
        """
        ...
    


