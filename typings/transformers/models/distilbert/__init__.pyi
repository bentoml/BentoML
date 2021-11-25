

from typing import TYPE_CHECKING

from ...file_utils import (
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)
from .configuration_distilbert import (
    DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    DistilBertConfig,
    DistilBertOnnxConfig,
)
from .tokenization_distilbert import DistilBertTokenizer

_import_structure = ...
if is_tokenizers_available():
    ...
if is_torch_available():
    ...
if is_tf_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
