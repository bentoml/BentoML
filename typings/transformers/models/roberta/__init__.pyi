

from typing import TYPE_CHECKING

from ...file_utils import (
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)
from .configuration_roberta import (
    ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    RobertaConfig,
    RobertaOnnxConfig,
)
from .tokenization_roberta import RobertaTokenizer

_import_structure = ...
if is_tokenizers_available():
    ...
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
