

import typing
from typing import Text, Tuple

import six

from .base import FS
from .wrapfs import WrapFS

"""Manage a directory in a *parent* filesystem.
"""

_F = typing.TypeVar("_F", bound="FS", covariant=True)

@six.python_2_unicode_compatible
class SubFS(WrapFS[_F], typing.Generic[_F]):
    """A sub-directory on another filesystem.

    A SubFS is a filesystem object that maps to a sub-directory of
    another filesystem. This is the object that is returned by
    `~fs.base.FS.opendir`.

    """

    def __init__(self, parent_fs: _F, path: Text) -> None: ...
    def __repr__(self) -> Text: ...
    def __str__(self) -> Text: ...
    def delegate_fs(self) -> _F: ...
    def delegate_path(self, path: Text) -> Tuple[_F, Text]: ...

class ClosingSubFS(SubFS[_F], typing.Generic[_F]):
    """A version of `SubFS` which closes its parent when closed."""

    def close(self) -> None: ...
