

from typing import TYPE_CHECKING

from ...file_utils import _LazyModule, is_tf_available, is_torch_available
from .configuration_transfo_xl import (
    TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    TransfoXLConfig,
)
from .tokenization_transfo_xl import TransfoXLCorpus, TransfoXLTokenizer

_import_structure = ...
if is_torch_available():
    ...
if is_tf_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
