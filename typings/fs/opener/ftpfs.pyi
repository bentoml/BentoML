import typing
from typing import Text, Union
from ..errors import CreateFailed
from ..ftpfs import FTPFS
from ..subfs import SubFS
from .base import Opener
from .parse import ParseResult
from .registry import registry

if typing.TYPE_CHECKING: ...

@registry.install
class FTPOpener(Opener):
    protocols = ...
    @CreateFailed.catch_all
    def open_fs(
        self,
        fs_url: Text,
        parse_result: ParseResult,
        writeable: bool,
        create: bool,
        cwd: Text,
    ) -> Union[FTPFS, SubFS[FTPFS]]: ...
