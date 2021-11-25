

import typing
from typing import IO, Any, BinaryIO, Collection, Iterator, List, Optional, Text, Tuple

from .base import FS
from .enums import ResourceType
from .info import Info, RawInfo
from .permissions import Permissions
from .subfs import SubFS

"""Manage several filesystems through a single view.
"""
if typing.TYPE_CHECKING:
    M = ...
_PrioritizedFS = ...

class MultiFS(FS):
    """A filesystem that delegates to a sequence of other filesystems.

    Operations on the MultiFS will try each 'child' filesystem in order,
    until it succeeds. In effect, creating a filesystem that combines
    the files and dirs of its children.

    """

    _meta = ...
    def __init__(self, auto_close: bool = ...) -> None:
        """Create a new MultiFS.

        Arguments:
            auto_close (bool): If `True` (the default), the child
                filesystems will be closed when `MultiFS` is closed.

        """
        ...
    def __repr__(self) -> Text: ...
    def __str__(self) -> Text: ...
    def add_fs(
        self, name: Text, fs: FS, write: bool = ..., priority: int = ...
    ) -> None:
        """Add a filesystem to the MultiFS.

        Arguments:
            name (str): A unique name to refer to the filesystem being
                added.
            fs (FS or str): The filesystem (instance or URL) to add.
            write (bool): If this value is True, then the ``fs`` will
                be used as the writeable FS (defaults to False).
            priority (int): An integer that denotes the priority of the
                filesystem being added. Filesystems will be searched in
                descending priority order and then by the reverse order
                they were added. So by default, the most recently added
                filesystem will be looked at first.

        """
        ...
    def get_fs(self, name: Text) -> FS:
        """Get a filesystem from its name.

        Arguments:
            name (str): The name of a filesystem previously added.

        Returns:
            FS: the filesystem added as ``name`` previously.

        Raises:
            KeyError: If no filesystem with given ``name`` could be found.

        """
        ...
    def iterate_fs(self) -> Iterator[Tuple[Text, FS]]:
        """Get iterator that returns (name, fs) in priority order."""
        ...
    def which(
        self, path: Text, mode: Text = ...
    ) -> Tuple[Optional[Text], Optional[FS]]:
        """Get a tuple of (name, fs) that the given path would map to.

        Arguments:
            path (str): A path on the filesystem.
            mode (str): An `io.open` mode.

        """
        ...
    def close(self) -> None: ...
    def getinfo(
        self, path: Text, namespaces: Optional[Collection[Text]] = ...
    ) -> Info: ...
    def listdir(self, path: Text) -> List[Text]: ...
    def makedir(
        self: M,
        path: Text,
        permissions: Optional[Permissions] = ...,
        recreate: bool = ...,
    ) -> SubFS[FS]: ...
    def openbin(
        self, path: Text, mode: Text = ..., buffering: int = ..., **options: Any
    ) -> BinaryIO: ...
    def remove(self, path: Text) -> None: ...
    def removedir(self, path: Text) -> None: ...
    def scandir(
        self,
        path: Text,
        namespaces: Optional[Collection[Text]] = ...,
        page: Optional[Tuple[int, int]] = ...,
    ) -> Iterator[Info]: ...
    def readbytes(self, path: Text) -> bytes: ...
    def download(
        self,
        path: Text,
        file: BinaryIO,
        chunk_size: Optional[int] = ...,
        **options: Any
    ) -> None: ...
    def readtext(
        self,
        path: Text,
        encoding: Optional[Text] = ...,
        errors: Optional[Text] = ...,
        newline: Text = ...,
    ) -> Text: ...
    def getsize(self, path: Text) -> int: ...
    def getsyspath(self, path: Text) -> Text: ...
    def gettype(self, path: Text) -> ResourceType: ...
    def geturl(self, path: Text, purpose: Text = ...) -> Text: ...
    def hassyspath(self, path: Text) -> bool: ...
    def hasurl(self, path: Text, purpose: Text = ...) -> bool: ...
    def isdir(self, path: Text) -> bool: ...
    def isfile(self, path: Text) -> bool: ...
    def setinfo(self, path: Text, info: RawInfo) -> None: ...
    def validatepath(self, path: Text) -> Text: ...
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
        **kwargs: Any
    ) -> IO: ...
    def upload(
        self,
        path: Text,
        file: BinaryIO,
        chunk_size: Optional[int] = ...,
        **options: Any
    ) -> None: ...
    def writebytes(self, path: Text, contents: bytes) -> None: ...
    def writetext(
        self,
        path: Text,
        contents: Text,
        encoding: Text = ...,
        errors: Optional[Text] = ...,
        newline: Text = ...,
    ) -> None: ...
