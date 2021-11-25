

import typing
from typing import Text

from ..tarfs import TarFS
from .base import Opener
from .parse import ParseResult
from .registry import registry

"""`TarFS` opener definition.
"""
if typing.TYPE_CHECKING: ...

@registry.install
class TarOpener(Opener):
    """`TarFS` opener."""

    protocols = ...
    def open_fs(
        self,
        fs_url: Text,
        parse_result: ParseResult,
        writeable: bool,
        create: bool,
        cwd: Text,
    ) -> TarFS: ...
