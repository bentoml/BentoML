from typing import TYPE_CHECKING, Union
import numpy as np
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from .base import Pipeline

if TYPE_CHECKING: ...
logger = ...

def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array: ...

class AutomaticSpeechRecognitionPipeline(Pipeline):
    def __init__(
        self, feature_extractor: SequenceFeatureExtractor, *args, **kwargs
    ) -> None: ...
