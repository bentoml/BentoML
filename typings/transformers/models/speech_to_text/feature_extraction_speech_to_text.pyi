

from typing import List, Optional, Union

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...file_utils import PaddingStrategy, TensorType

"""
Feature extractor class for Speech2Text
"""
logger = ...
class Speech2TextFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Speech2Text feature extractor.

    This feature extractor inherits from :class:`~transformers.Speech2TextFeatureExtractor` which contains most of the
    main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech using TorchAudio and applies utterance-level cepstral
    mean and variance normalization to the extracted features.

    Args:
        feature_size (:obj:`int`, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (:obj:`int`, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in Hertz per second (Hz).
        num_mel_bins (:obj:`int`, defaults to 80):
            Number of Mel-frequency bins.
        padding_value (:obj:`float`, defaults to 0.0):
            The value that is used to fill the padding vectors.
        do_ceptral_normalize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to apply utterance-level cepstral mean and variance normalization to extracted features.
        normalize_means (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to zero-mean normalize the extracted features.
        normalize_vars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to unit-variance normalize the extracted features.
    """
    model_input_names = ...
    def __init__(self, feature_size=..., sampling_rate=..., num_mel_bins=..., padding_value=..., do_ceptral_normalize=..., normalize_means=..., normalize_vars=..., **kwargs) -> None:
        ...
    
    @staticmethod
    def utterance_cmvn(x: np.ndarray, normalize_means: Optional[bool] = ..., normalize_vars: Optional[bool] = ...) -> np.ndarray:
        ...
    
    def normalize(self, input_values: List[np.ndarray]) -> List[np.ndarray]:
        ...
    
    def __call__(self, raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]], padding: Union[bool, str, PaddingStrategy] = ..., max_length: Optional[int] = ..., pad_to_multiple_of: Optional[int] = ..., return_tensors: Optional[Union[str, TensorType]] = ..., sampling_rate: Optional[int] = ..., return_attention_mask: Optional[bool] = ..., **kwargs) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s). sequences.

        Args:
            raw_speech (:obj:`np.ndarray`, :obj:`List[float]`, :obj:`List[np.ndarray]`, :obj:`List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
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

                    For Speech2TextTransoformer models, :obj:`attention_mask` should alwys be passed for batched
                    inference, to avoid subtle bugs.

            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
            sampling_rate (:obj:`int`, `optional`):
                The sampling rate at which the :obj:`raw_speech` input was sampled. It is strongly recommended to pass
                :obj:`sampling_rate` at the forward call to prevent silent errors.
            padding_value (:obj:`float`, defaults to 0.0):
                The value that is used to fill the padding values / vectors.
        """
        ...
    


