

from typing import Dict, List, Optional, Union

from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from .file_utils import PaddingStrategy, TensorType

logger = ...
class SequenceFeatureExtractor(FeatureExtractionMixin):
    """
    This is a general feature extraction class for speech recognition.

    Args:
        feature_size (:obj:`int`):
            The feature dimension of the extracted features.
        sampling_rate (:obj:`int`):
            The sampling rate at which the audio files should be digitalized expressed in Hertz per second (Hz).
        padding_value (:obj:`float`):
            The value that is used to fill the padding values / vectors.
    """
    def __init__(self, feature_size: int, sampling_rate: int, padding_value: float, **kwargs) -> None:
        ...
    
    def pad(self, processed_features: Union[BatchFeature, List[BatchFeature], Dict[str, BatchFeature], Dict[str, List[BatchFeature]], List[Dict[str, BatchFeature]]], , padding: Union[bool, str, PaddingStrategy] = ..., max_length: Optional[int] = ..., pad_to_multiple_of: Optional[int] = ..., return_attention_mask: Optional[bool] = ..., return_tensors: Optional[Union[str, TensorType]] = ...) -> BatchFeature:
        """
        Pad input values / input vectors or a batch of input values / input vectors up to predefined length or to the
        max sequence length in the batch.

        Padding side (left/right) padding values are defined at the feature extractor level (with
        ``self.padding_side``, ``self.padding_value``)

        .. note::

            If the ``processed_features`` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors,
            the result will use the same type unless you provide a different tensor type with ``return_tensors``. In
            the case of PyTorch tensors, you will lose the specific device of your tensors however.

        Args:
            processed_features (:class:`~transformers.BatchFeature`, list of :class:`~transformers.BatchFeature`, :obj:`Dict[str, List[float]]`, :obj:`Dict[str, List[List[float]]` or :obj:`List[Dict[str, List[float]]]`):
                Processed inputs. Can represent one input (:class:`~transformers.BatchFeature` or :obj:`Dict[str,
                List[float]]`) or a batch of input values / vectors (list of :class:`~transformers.BatchFeature`,
                `Dict[str, List[List[float]]]` or `List[Dict[str, List[float]]]`) so you can use this method during
                preprocessing as well as in a PyTorch Dataloader collate function.

                Instead of :obj:`List[float]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow
                tensors), see the note above for the return type.
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
            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
        """
        ...
    


