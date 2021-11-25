

from typing import TYPE_CHECKING

from ...file_utils import (
    _LazyModule,
    is_sentencepiece_available,
    is_speech_available,
    is_torch_available,
)
from .configuration_speech_to_text import (
    SPEECH_TO_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    Speech2TextConfig,
)

_import_structure = ...
if is_sentencepiece_available():
    ...
if is_speech_available():
    ...
if is_torch_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
