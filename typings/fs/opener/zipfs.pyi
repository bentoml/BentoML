"""
This type stub file was generated by pyright.
"""

import typing
from .base import Opener
from .registry import registry
from typing import Text
from .parse import ParseResult
from ..zipfs import ZipFS

"""`ZipFS` opener definition.
"""
if typing.TYPE_CHECKING: ...

@registry.install
class ZipOpener(Opener):
    """`ZipFS` opener."""

    protocols = ...
    def open_fs(
        self,
        fs_url: Text,
        parse_result: ParseResult,
        writeable: bool,
        create: bool,
        cwd: Text,
    ) -> ZipFS: ...
