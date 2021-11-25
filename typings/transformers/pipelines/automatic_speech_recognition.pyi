

from typing import TYPE_CHECKING, Union

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from .base import Pipeline

if TYPE_CHECKING:
    ...
logger = ...
def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array:
    """
    Helper function to read an audio file through ffmpeg.
    """
    ...

class AutomaticSpeechRecognitionPipeline(Pipeline):
    """
    Pipeline that aims at extracting spoken text contained within some audio.

    The input can be either a raw waveform or a audio file. In case of the audio file, ffmpeg should be installed for
    to support multiple audio formats
    """
    def __init__(self, feature_extractor: SequenceFeatureExtractor, *args, **kwargs) -> None:
        """
        Arguments:
            feature_extractor (:obj:`~transformers.SequenceFeatureExtractor`):
                The feature extractor that will be used by the pipeline to encode waveform for the model.
            model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
                The model that will be used by the pipeline to make predictions. This needs to be a model inheriting
                from :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel`
                for TensorFlow.
            tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
                The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
                :class:`~transformers.PreTrainedTokenizer`.
            modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`):
                Model card attributed to the model for this pipeline.
            framework (:obj:`str`, `optional`):
                The framework to use, either :obj:`"pt"` for PyTorch or :obj:`"tf"` for TensorFlow. The specified
                framework must be installed.

                If no framework is specified, will default to the one currently installed. If no framework is specified
                and both frameworks are installed, will default to the framework of the :obj:`model`, or to PyTorch if
                no model is provided.
            device (:obj:`int`, `optional`, defaults to -1):
                Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the
                model on the associated CUDA device id.
        """
        ...
    
    def __call__(self, inputs: Union[np.ndarray, bytes, str], **kwargs):
        """
        Classify the sequence(s) given as inputs. See the :obj:`~transformers.AutomaticSpeechRecognitionPipeline`
        documentation for more information.

        Args:
            inputs (:obj:`np.ndarray` or :obj:`bytes` or :obj:`str`):
                The inputs is either a raw waveform (:obj:`np.ndarray` of shape (n, ) of type :obj:`np.float32` or
                :obj:`np.float64`) at the correct sampling rate (no further check will be done) or a :obj:`str` that is
                the filename of the audio file, the file will be read at the correct sampling rate to get the waveform
                using `ffmpeg`. This requires `ffmpeg` to be installed on the system. If `inputs` is :obj:`bytes` it is
                supposed to be the content of an audio file and is interpreted by `ffmpeg` in the same way.

        Return:
            A :obj:`dict` with the following keys:

            - **text** (:obj:`str`) -- The recognized text.
        """
        ...
    


