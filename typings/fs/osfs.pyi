from __future__ import annotations

from typing import IO
from typing import Any
from typing import BinaryIO
from typing import Collection
from typing import Iterator
from typing import List
from typing import Optional
from typing import SupportsInt
from typing import Text
from typing import Tuple

from .base import FS
from .enums import ResourceType
from .info import Info
from .info import RawInfo
from .permissions import Permissions
from .subfs import SubFS

_O = Any

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

    STAT_TO_RESOURCE_TYPE: dict[str, Any] = ...

    def getinfo(
        self, path: Text, namespaces: Optional[Collection[Text]] = ...
    ) -> Info: ...
    def listdir(self, path: Text) -> List[Text]: ...
    def makedir(
        self,
        path: Text,
        permissions: Optional[Permissions] = ...,
        recreate: bool = ...,
    ) -> SubFS[_O]: ...
    def openbin(
        self, path: Text, mode: Text = ..., buffering: int = ..., **options: Any
    ) -> BinaryIO: ...
    def remove(self, path: Text) -> None: ...
    def removedir(self, path: Text) -> None: ...
    def copy(self, src_path: Text, dst_path: Text, overwrite: bool = ...) -> None: ...
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
        **options: Any,
    ) -> IO[Any]: ...
    def setinfo(self, path: Text, info: RawInfo) -> None: ...
    def validatepath(self, path: Text) -> Text: ...
