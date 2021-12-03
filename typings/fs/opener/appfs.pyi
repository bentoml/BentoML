"""
This type stub file was generated by pyright.
"""

import typing
from .base import Opener
from .registry import registry
from typing import Text, Union
from .parse import ParseResult
from ..appfs import _AppFS
from ..subfs import SubFS

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
