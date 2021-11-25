

from typing import TYPE_CHECKING

from ...file_utils import _LazyModule, is_tf_available, is_torch_available
from .configuration_blenderbot import (
    BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    BlenderbotConfig,
)
from .tokenization_blenderbot import BlenderbotTokenizer

_import_structure = ...
if is_torch_available():
    ...
if is_tf_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
