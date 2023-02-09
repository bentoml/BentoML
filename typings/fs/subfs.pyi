import typing
from typing import Text
from typing import Tuple

import six

from .base import FS
from .wrapfs import WrapFS

_F = typing.TypeVar("_F", bound="FS", covariant=True)

@six.python_2_unicode_compatible
class SubFS(WrapFS[_F], typing.Generic[_F]):
    def __init__(self, parent_fs: _F, path: Text) -> None: ...
    def __repr__(self) -> Text: ...
    def __str__(self) -> Text: ...
    def delegate_fs(self) -> _F: ...
    def delegate_path(self, path: Text) -> Tuple[_F, Text]: ...

class ClosingSubFS(SubFS[_F], typing.Generic[_F]):
    def close(self) -> None: ...
