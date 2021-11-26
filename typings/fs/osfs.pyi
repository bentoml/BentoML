import sys
import typing
from typing import (
    IO,
    Any,
    BinaryIO,
    Collection,
    Iterator,
    List,
    Optional,
    SupportsInt,
    Text,
    Tuple,
)
import six
from .base import FS, _OpendirFactory
from .enums import ResourceType
from .info import Info, RawInfo
from .permissions import Permissions
from .subfs import SubFS

if typing.TYPE_CHECKING:
    _O = ...
log = ...
_WINDOWS_PLATFORM = ...

@six.python_2_unicode_compatible
class OSFS(FS):
    def __init__(
        self,
        root_path: Text,
        create: bool = ...,
        create_mode: SupportsInt = ...,
        expand_vars: bool = ...,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    STAT_TO_RESOURCE_TYPE = ...
    def getinfo(
        self, path: Text, namespaces: Optional[Collection[Text]] = ...
    ) -> Info: ...
    def listdir(self, path: Text) -> List[Text]: ...
    def makedir(
        self: _O,
        path: Text,
        permissions: Optional[Permissions] = ...,
        recreate: bool = ...,
    ) -> SubFS[_O]: ...
    def openbin(
        self, path: Text, mode: Text = ..., buffering: int = ..., **options: Any
    ) -> BinaryIO: ...
    def remove(self, path: Text) -> None: ...
    def removedir(self, path: Text) -> None: ...
    if typing.TYPE_CHECKING:
        def opendir(
            self: _O, path: Text, factory: Optional[_OpendirFactory] = ...
        ) -> SubFS[_O]: ...
    if sys.version_info[:2] < (3, 8) and sendfile is not None:
        _sendfile_error_codes = ...
        def copy(
            self, src_path: Text, dst_path: Text, overwrite: bool = ...
        ) -> None: ...
    else:
        def copy(
            self, src_path: Text, dst_path: Text, overwrite: bool = ...
        ) -> None: ...
    if scandir: ...
    else: ...
    def scandir(
        self,
        path: Text,
        namespaces: Optional[Collection[Text]] = ...,
        page: Optional[Tuple[int, int]] = ...,
    ) -> Iterator[Info]: ...
    def getsyspath(self, path: Text) -> Text: ...
    def geturl(self, path: Text, purpose: Text = ...) -> Text: ...
    def gettype(self, path: Text) -> ResourceType: ...
    def islink(self, path: Text) -> bool: ...
    def open(
        self,
        path: Text,
        mode: Text = ...,
        buffering: int = ...,
        encoding: Optional[Text] = ...,
        errors: Optional[Text] = ...,
        newline: Text = ...,
        line_buffering: bool = ...,
        **options: Any
    ) -> IO: ...
    def setinfo(self, path: Text, info: RawInfo) -> None: ...
    def validatepath(self, path: Text) -> Text: ...
