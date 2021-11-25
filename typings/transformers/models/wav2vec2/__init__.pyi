

from typing import TYPE_CHECKING

from ...file_utils import (
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)
from .configuration_wav2vec2 import (
    WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP,
    Wav2Vec2Config,
)
from .feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor
from .processing_wav2vec2 import Wav2Vec2Processor
from .tokenization_wav2vec2 import Wav2Vec2CTCTokenizer, Wav2Vec2Tokenizer

_import_structure = ...
if is_torch_available():
    ...
if is_tf_available():
    ...
if is_flax_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
