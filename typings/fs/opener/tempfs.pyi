from typing import Text

from ..tempfs import TempFS
from .base import Opener
from .parse import ParseResult
from .registry import registry

@registry.install
class TempOpener(Opener):
    protocols = ...

    def open_fs(
        self,
        fs_url: Text,
        parse_result: ParseResult,
        writeable: bool,
        create: bool,
        cwd: Text,
    ) -> TempFS: ...
