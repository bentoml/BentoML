

from typing import TYPE_CHECKING

from ...file_utils import _LazyModule, is_flax_available, is_torch_available
from .configuration_gpt_neo import (
    GPT_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP,
    GPTNeoConfig,
    GPTNeoOnnxConfig,
)

_import_structure = ...
if is_torch_available():
    ...
if is_flax_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
