from typing import Text

from ..osfs import OSFS
from .base import Opener
from .parse import ParseResult
from .registry import registry

@registry.install
class OSFSOpener(Opener):
    protocols = ...
    def open_fs(
        self,
        fs_url: Text,
        parse_result: ParseResult,
        writeable: bool,
        create: bool,
        cwd: Text,
    ) -> OSFS: ...
