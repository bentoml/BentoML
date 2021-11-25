

from typing import TYPE_CHECKING

from ...file_utils import (
    _LazyModule,
    is_flax_available,
    is_torch_available,
    is_vision_available,
)
from .configuration_vit import VIT_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTConfig

_import_structure = ...
if is_vision_available():
    ...
if is_torch_available():
    ...
if is_flax_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
