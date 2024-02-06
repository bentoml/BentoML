import abc
import typing
from datetime import datetime
from threading import RLock
from types import TracebackType
from typing import IO
from typing import Any
from typing import AnyStr
from typing import BinaryIO
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Text
from typing import Tuple
from typing import Type
from typing import Union

import six

from .enums import ResourceType
from .glob import BoundGlobber
from .info import Info
from .info import RawInfo
from .permissions import Permissions
from .subfs import SubFS
from .walk import BoundWalker
from .walk import Walker

_F = typing.TypeVar("_F", bound="FS")
_T = typing.TypeVar("_T", bound="FS")
_OpendirFactory = Callable[[_T, Text], SubFS[_T]]
__all__ = ["FS"]

@six.add_metaclass(abc.ABCMeta)
class FS:
    _meta: Dict[Text, Union[Text, int, bool, None]] = ...
    walker_class = Walker
    subfs_class = None

    def __init__(self) -> None: ...
    def __del__(self) -> None: ...
    def __enter__(self) -> FS: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None: ...
    @property
    def glob(self) -> BoundGlobber: ...
    @property
    def walk(self: _F) -> BoundWalker[_F]: ...
    @abc.abstractmethod
    def getinfo(
        self, path: Text, namespaces: Optional[Collection[Text]] = ...
    ) -> Info: ...
    @abc.abstractmethod
    def listdir(self, path: Text) -> List[Text]: ...
    @abc.abstractmethod
    def makedir(
        self, path: Text, permissions: Optional[Permissions] = ..., recreate: bool = ...
    ) -> SubFS[FS]: ...
    @abc.abstractmethod
    def openbin(
        self, path: Text, mode: Text = ..., buffering: int = ..., **options: Any
    ) -> BinaryIO: ...
    @abc.abstractmethod
    def remove(self, path: Text) -> None: ...
    @abc.abstractmethod
    def removedir(self, path: Text) -> None: ...
    @abc.abstractmethod
    def setinfo(self, path: Text, info: RawInfo) -> None: ...
    def appendbytes(self, path: Text, data: bytes) -> None: ...
    def appendtext(
        self,
        path: Text,
        text: Text,
        encoding: Text = ...,
        errors: Optional[Text] = ...,
        newline: Text = ...,
    ) -> None: ...
    def close(self) -> None: ...
    def copy(self, src_path: Text, dst_path: Text, overwrite: bool = ...) -> None: ...
    def copydir(self, src_path: Text, dst_path: Text, create: bool = ...) -> None: ...
    def create(self, path: Text, wipe: bool = ...) -> bool: ...
    def desc(self, path: Text) -> Text: ...
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

    getbytes = readbytes

    def download(
        self,
        path: Text,
        file: BinaryIO,
        chunk_size: Optional[int] = ...,
        **options: Any,
    ) -> None: ...

    getfile = download

    def readtext(
        self,
        path: Text,
        encoding: Optional[Text] = ...,
        errors: Optional[Text] = ...,
        newline: Text = ...,
    ) -> Text: ...

    gettext = readtext

    def getmeta(self, namespace: Text = ...) -> Mapping[Text, object]: ...
    def getsize(self, path: Text) -> int: ...
    def getsyspath(self, path: Text) -> Text: ...
    def getospath(self, path: Text) -> bytes: ...
    def gettype(self, path: Text) -> ResourceType: ...
    def geturl(self, path: Text, purpose: Text = ...) -> Text: ...
    def hassyspath(self, path: Text) -> bool: ...
    def hasurl(self, path: Text, purpose: Text = ...) -> bool: ...
    def isclosed(self) -> bool: ...
    def isdir(self, path: Text) -> bool: ...
    def isempty(self, path: Text) -> bool: ...
    def isfile(self, path: Text) -> bool: ...
    def islink(self, path: Text) -> bool: ...
    def lock(self) -> RLock: ...
    def movedir(self, src_path: Text, dst_path: Text, create: bool = ...) -> None: ...
    def makedirs(
        self, path: Text, permissions: Optional[Permissions] = ..., recreate: bool = ...
    ) -> SubFS[FS]: ...
    def move(self, src_path: Text, dst_path: Text, overwrite: bool = ...) -> None: ...
    def open(
        self,
        path: Text,
        mode: Text = ...,
        buffering: int = ...,
        encoding: Optional[Text] = ...,
        errors: Optional[Text] = ...,
        newline: Text = ...,
        **options: Any,
    ) -> IO[Any]: ...
    def opendir(
        self: _F, path: Text, factory: Optional[_OpendirFactory[_F]] = ...
    ) -> SubFS[FS]: ...
    def removetree(self, dir_path: Text) -> None: ...
    def scandir(
        self,
        path: Text,
        namespaces: Optional[Collection[Text]] = ...,
        page: Optional[Tuple[int, int]] = ...,
    ) -> Iterator[Info]: ...
    def writebytes(self, path: Text, contents: bytes) -> None: ...

    setbytes = writebytes

    def upload(
        self,
        path: Text,
        file: BinaryIO,
        chunk_size: Optional[int] = ...,
        **options: Any,
    ) -> None: ...

    setbinfile = upload

    def writefile(
        self,
        path: Text,
        file: IO[AnyStr],
        encoding: Optional[Text] = ...,
        errors: Optional[Text] = ...,
        newline: Text = ...,
    ) -> None: ...

    setfile = writefile

    def settimes(
        self,
        path: Text,
        accessed: Optional[datetime] = ...,
        modified: Optional[datetime] = ...,
    ) -> None: ...
    def writetext(
        self,
        path: Text,
        contents: Text,
        encoding: Text = ...,
        errors: Optional[Text] = ...,
        newline: Text = ...,
    ) -> None: ...

    settext = writetext

    def touch(self, path: Text) -> None: ...
    def validatepath(self, path: Text) -> Text: ...
    def getbasic(self, path: Text) -> Info: ...
    def getdetails(self, path: Text) -> Info: ...
    def check(self) -> None: ...
    def match(self, patterns: Optional[Iterable[Text]], name: Text) -> bool: ...
    def tree(self, **kwargs: Any) -> None: ...
    def hash(self, path: Text, name: Text) -> Text: ...
