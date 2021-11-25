

from typing import TYPE_CHECKING

from ...file_utils import _LazyModule, is_tf_available, is_torch_available
from .configuration_ctrl import CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP, CTRLConfig
from .tokenization_ctrl import CTRLTokenizer

_import_structure = ...
if is_torch_available():
    ...
if is_tf_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
