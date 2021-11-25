

import typing
from datetime import datetime
from threading import RLock
from typing import (
    IO,
    Any,
    AnyStr,
    BinaryIO,
    Callable,
    Collection,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Text,
    Tuple,
)

import six

from .base import FS
from .enums import ResourceType
from .info import Info, RawInfo
from .permissions import Permissions
from .subfs import SubFS
from .walk import BoundWalker

"""Base class for filesystem wrappers.
"""
_T = typing.TypeVar("_T", bound="FS")
_OpendirFactory = Callable[[_T, Text], SubFS[_T]]

_F = typing.TypeVar("_F", bound="FS", covariant=True)
_W = typing.TypeVar("_W", bound="WrapFS[FS]")

@six.python_2_unicode_compatible
class WrapFS(FS, typing.Generic[_F]):
    """A proxy for a filesystem object.

    This class exposes an filesystem interface, where the data is
    stored on another filesystem(s), and is the basis for
    `~fs.subfs.SubFS` and other *virtual* filesystems.

    """

    wrap_name: Optional[Text] = ...
    def __init__(self, wrap_fs: _F) -> None: ...
    def __repr__(self) -> Text: ...
    def __str__(self) -> Text: ...
    def delegate_path(self, path: Text) -> Tuple[_F, Text]:
        """Encode a path for proxied filesystem.

        Arguments:
            path (str): A path on the filesystem.

        Returns:
            (FS, str): a tuple of ``(<filesystem>, <new_path>)``

        """
        ...
    def delegate_fs(self) -> _F:
        """Get the proxied filesystem.

        This method should return a filesystem for methods not
        associated with a path, e.g. `~fs.base.FS.getmeta`.

        """
        ...
    def appendbytes(self, path: Text, data: bytes) -> None: ...
    def appendtext(
        self,
        path: Text,
        text: Text,
        encoding: Text = ...,
        errors: Optional[Text] = ...,
        newline: Text = ...,
    ) -> None: ...
    def getinfo(
        self, path: Text, namespaces: Optional[Collection[Text]] = ...
    ) -> Info: ...
    def listdir(self, path: Text) -> List[Text]: ...
    def lock(self) -> RLock: ...
    def makedir(
        self, path: Text, permissions: Optional[Permissions] = ..., recreate: bool = ...
    ) -> SubFS[FS]: ...
    def move(self, src_path: Text, dst_path: Text, overwrite: bool = ...) -> None: ...
    def movedir(self, src_path: Text, dst_path: Text, create: bool = ...) -> None: ...
    def openbin(
        self, path: Text, mode: Text = ..., buffering: int = ..., **options: Any
    ) -> BinaryIO: ...
    def remove(self, path: Text) -> None: ...
    def removedir(self, path: Text) -> None: ...
    def removetree(self, dir_path: Text) -> None: ...
    def scandir(
        self,
        path: Text,
        namespaces: Optional[Collection[Text]] = ...,
        page: Optional[Tuple[int, int]] = ...,
    ) -> Iterator[Info]: ...
    def setinfo(self, path: Text, info: RawInfo) -> None: ...
    def settimes(
        self,
        path: Text,
        accessed: Optional[datetime] = ...,
        modified: Optional[datetime] = ...,
    ) -> None: ...
    def touch(self, path: Text) -> None: ...
    def copy(self, src_path: Text, dst_path: Text, overwrite: bool = ...) -> None: ...
    def copydir(self, src_path: Text, dst_path: Text, create: bool = ...) -> None: ...
    def create(self, path: Text, wipe: bool = ...) -> bool: ...
    def desc(self, path: Text) -> Text: ...
    def download(
        self,
        path: Text,
        file: BinaryIO,
        chunk_size: Optional[int] = ...,
        **options: Any
    ) -> None: ...
    def exists(self, path: Text) -> bool: ...
    def filterdir(
        self,
        path: Text,
        files: Optional[Iterable[Text]] = ...,
        dirs: Optional[Iterable[Text]] = ...,
        exclude_dirs: Optional[Iterable[Text]] = ...,
        exclude_files: Optional[Iterable[Text]] = ...,
        namespaces: Optional[Collection[Text]] = ...,
        page: Optional[Tuple[int, int]] = ...,
    ) -> Iterator[Info]: ...
    def readbytes(self, path: Text) -> bytes: ...
    def readtext(
        self,
        path: Text,
        encoding: Optional[Text] = ...,
        errors: Optional[Text] = ...,
        newline: Text = ...,
    ) -> Text: ...
    def getmeta(self, namespace: Text = ...) -> Mapping[Text, object]: ...
    def getsize(self, path: Text) -> int: ...
    def getsyspath(self, path: Text) -> Text: ...
    def gettype(self, path: Text) -> ResourceType: ...
    def geturl(self, path: Text, purpose: Text = ...) -> Text: ...
    def hassyspath(self, path: Text) -> bool: ...
    def hasurl(self, path: Text, purpose: Text = ...) -> bool: ...
    def isdir(self, path: Text) -> bool: ...
    def isfile(self, path: Text) -> bool: ...
    def islink(self, path: Text) -> bool: ...
    def makedirs(
        self, path: Text, permissions: Optional[Permissions] = ..., recreate: bool = ...
    ) -> SubFS[FS]: ...
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
    ) -> IO[AnyStr]: ...
    def opendir(
        self: _W, path: Text, factory: Optional[_OpendirFactory[_W]] = ...
    ) -> SubFS[_W]: ...
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
        file: IO[AnyStr],
        encoding: Optional[Text] = ...,
        errors: Optional[Text] = ...,
        newline: Text = ...,
    ) -> None: ...
    def validatepath(self, path: Text) -> Text: ...
    def hash(self, path: Text, name: Text) -> Text: ...
    @property
    def walk(self: _W) -> BoundWalker[_W]: ...
