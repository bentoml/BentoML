

from typing import TYPE_CHECKING

from ...file_utils import _LazyModule, is_torch_available
from .configuration_prophetnet import (
    PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ProphetNetConfig,
)
from .tokenization_prophetnet import ProphetNetTokenizer

_import_structure = ...
if is_torch_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
