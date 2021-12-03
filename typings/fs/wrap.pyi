import typing
from datetime import datetime
from typing import IO, Any, BinaryIO, Collection, Iterator, Optional, Text, Tuple
from .info import Info, RawInfo
from .permissions import Permissions
from .subfs import SubFS
from .wrapfs import WrapFS

if typing.TYPE_CHECKING: ...
_W = ...
_T = ...
_F = ...

def read_only(fs: _T) -> WrapReadOnly[_T]: ...
def cache_directory(fs: _T) -> WrapCachedDir[_T]: ...

class WrapCachedDir(WrapFS[_F], typing.Generic[_F]):
    wrap_name = ...
    def __init__(self, wrap_fs: _F) -> None: ...
    def scandir(
        self,
        path: Text,
        namespaces: Optional[Collection[Text]] = ...,
        page: Optional[Tuple[(int, int)]] = ...,
    ) -> Iterator[Info]: ...
    def getinfo(
        self, path: Text, namespaces: Optional[Collection[Text]] = ...
    ) -> Info: ...
    def isdir(self, path: Text) -> bool: ...
    def isfile(self, path: Text) -> bool: ...

class WrapReadOnly(WrapFS[_F], typing.Generic[_F]):
    wrap_name = ...
    def appendbytes(self, path: Text, data: bytes) -> None: ...
    def appendtext(
        self,
        path: Text,
        text: Text,
        encoding: Text = ...,
        errors: Optional[Text] = ...,
        newline: Text = ...,
    ) -> None: ...
    def makedir(
        self: _W,
        path: Text,
        permissions: Optional[Permissions] = ...,
        recreate: bool = ...,
    ) -> SubFS[_W]: ...
    def move(self, src_path: Text, dst_path: Text, overwrite: bool = ...) -> None: ...
    def openbin(
        self, path: Text, mode: Text = ..., buffering: int = ..., **options: Any
    ) -> BinaryIO: ...
    def remove(self, path: Text) -> None: ...
    def removedir(self, path: Text) -> None: ...
    def removetree(self, path: Text) -> None: ...
    def setinfo(self, path: Text, info: RawInfo) -> None: ...
    def writetext(
        self,
        path: Text,
        contents: Text,
        encoding: Text = ...,
        errors: Optional[Text] = ...,
        newline: Text = ...,
    ) -> None: ...
    def settimes(
        self,
        path: Text,
        accessed: Optional[datetime] = ...,
        modified: Optional[datetime] = ...,
    ) -> None: ...
    def copy(self, src_path: Text, dst_path: Text, overwrite: bool = ...) -> None: ...
    def create(self, path: Text, wipe: bool = ...) -> bool: ...
    def makedirs(
        self: _W,
        path: Text,
        permissions: Optional[Permissions] = ...,
        recreate: bool = ...,
    ) -> SubFS[_W]: ...
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
    def writebytes(self, path: Text, contents: bytes) -> None: ...
    def upload(
        self,
        path: Text,
        file: BinaryIO,
        chunk_size: Optional[int] = ...,
        **options: Any
    ) -> None: ...
    def writefile(
        self,
        path: Text,
        file: IO,
        encoding: Optional[Text] = ...,
        errors: Optional[Text] = ...,
        newline: Text = ...,
    ) -> None: ...
    def touch(self, path: Text) -> None: ...
