

import typing
from typing import Text

from ..memoryfs import MemoryFS
from .base import Opener
from .parse import ParseResult
from .registry import registry

"""`MemoryFS` opener definition.
"""
if typing.TYPE_CHECKING: ...

@registry.install
class MemOpener(Opener):
    """`MemoryFS` opener."""

    protocols = ...
    def open_fs(
        self,
        fs_url: Text,
        parse_result: ParseResult,
        writeable: bool,
        create: bool,
        cwd: Text,
    ) -> MemoryFS: ...
