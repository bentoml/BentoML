import typing
from typing import Text
from ..zipfs import ZipFS
from .base import Opener
from .parse import ParseResult
from .registry import registry

if typing.TYPE_CHECKING: ...

@registry.install
class ZipOpener(Opener):
    protocols = ...
    def open_fs(
        self,
        fs_url: Text,
        parse_result: ParseResult,
        writeable: bool,
        create: bool,
        cwd: Text,
    ) -> ZipFS: ...
