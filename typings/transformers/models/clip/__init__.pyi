

from typing import TYPE_CHECKING

from ...file_utils import (
    _LazyModule,
    is_flax_available,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)
from .configuration_clip import (
    CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
    CLIPConfig,
    CLIPTextConfig,
    CLIPVisionConfig,
)
from .tokenization_clip import CLIPTokenizer

_import_structure = ...
if is_tokenizers_available():
    ...
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
