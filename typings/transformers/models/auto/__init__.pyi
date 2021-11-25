

from typing import TYPE_CHECKING

from ...file_utils import (
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)
from .auto_factory import get_values
from .configuration_auto import (
    ALL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    CONFIG_MAPPING,
    MODEL_NAMES_MAPPING,
    AutoConfig,
)
from .feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING, AutoFeatureExtractor
from .tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer

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
