import typing
from typing import Optional, Text
import six
from .osfs import OSFS

if typing.TYPE_CHECKING: ...

@six.python_2_unicode_compatible
class TempFS(OSFS):
    def __init__(
        self,
        identifier: Text = ...,
        temp_dir: Optional[Text] = ...,
        auto_clean: bool = ...,
        ignore_clean_errors: bool = ...,
    ) -> None: ...
    def __repr__(self) -> Text: ...
    def __str__(self) -> Text: ...
    def close(self) -> None: ...
    def clean(self) -> None: ...
