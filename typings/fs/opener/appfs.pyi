

import typing
from typing import Text, Union

from ..appfs import _AppFS
from ..subfs import SubFS
from .base import Opener
from .parse import ParseResult
from .registry import registry

"""``AppFS`` opener definition.
"""
if typing.TYPE_CHECKING: ...

@registry.install
class AppFSOpener(Opener):
    """``AppFS`` opener."""

    protocols = ...
    _protocol_mapping = ...
    def open_fs(
        self,
        fs_url: Text,
        parse_result: ParseResult,
        writeable: bool,
        create: bool,
        cwd: Text,
    ) -> Union[_AppFS, SubFS[_AppFS]]: ...
