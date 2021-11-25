

import array
import io
import mmap
import socket
import typing
from contextlib import contextmanager
from ftplib import FTP
from typing import (
    Any,
    BinaryIO,
    ByteString,
    Container,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    SupportsInt,
    Text,
    Tuple,
    Union,
)

from six import PY2

from .base import FS, _OpendirFactory
from .info import Info, RawInfo
from .permissions import Permissions
from .subfs import SubFS

"""Manage filesystems on remote FTP servers.
"""
if typing.TYPE_CHECKING: ...
_F = ...
__all__ = ["FTPFS"]

@contextmanager
def ftp_errors(fs: FTPFS, path: Optional[Text] = ...) -> Iterator[None]: ...
@contextmanager
def manage_ftp(ftp: FTP) -> Iterator[FTP]: ...

if PY2: ...
else: ...

class FTPFile(io.RawIOBase):
    def __init__(self, ftpfs: FTPFS, path: Text, mode: Text) -> None: ...
    @property
    def read_conn(self) -> socket.socket: ...
    @property
    def write_conn(self) -> socket.socket: ...
    def __repr__(self) -> str: ...
    def close(self) -> None: ...
    def tell(self) -> int: ...
    def readable(self) -> bool: ...
    def read(self, size: int = ...) -> bytes: ...
    def readinto(
        self, buffer: Union[bytearray, memoryview, array.array[Any], mmap.mmap]
    ) -> int: ...
    def readline(self, size: Optional[int] = ...) -> bytes: ...
    def readlines(self, hint: int = ...) -> List[bytes]: ...
    def writable(self) -> bool: ...
    def write(
        self, data: Union[bytes, memoryview, array.array[Any], mmap.mmap]
    ) -> int: ...
    def writelines(
        self, lines: Iterable[Union[bytes, memoryview, array.array[Any], mmap.mmap]]
    ) -> None: ...
    def truncate(self, size: Optional[int] = ...) -> int: ...
    def seekable(self) -> bool: ...
    def seek(self, pos: int, whence: SupportsInt = ...) -> int: ...

class FTPFS(FS):
    """A FTP (File Transport Protocol) Filesystem.

    Optionally, the connection can be made securely via TLS. This is known as
    FTPS, or FTP Secure. TLS will be enabled when using the ftps:// protocol,
    or when setting the `tls` argument to True in the constructor.

    Examples:
        Create with the constructor::

            >>> from fs.ftpfs import FTPFS
            >>> ftp_fs = FTPFS("demo.wftpserver.com")

        Or via an FS URL::

            >>> ftp_fs = fs.open_fs('ftp://test.rebex.net')

        Or via an FS URL, using TLS::

            >>> ftp_fs = fs.open_fs('ftps://demo.wftpserver.com')

        You can also use a non-anonymous username, and optionally a
        password, even within a FS URL::

            >>> ftp_fs = FTPFS("test.rebex.net", user="demo", passwd="password")
            >>> ftp_fs = fs.open_fs('ftp://demo:password@test.rebex.net')

        Connecting via a proxy is supported. If using a FS URL, the proxy
        URL will need to be added as a URL parameter::

            >>> ftp_fs = FTPFS("ftp.ebi.ac.uk", proxy="test.rebex.net")
            >>> ftp_fs = fs.open_fs('ftp://ftp.ebi.ac.uk/?proxy=test.rebex.net')

    """

    _meta = ...
    def __init__(
        self,
        host: Text,
        user: Text = ...,
        passwd: Text = ...,
        acct: Text = ...,
        timeout: int = ...,
        port: int = ...,
        proxy: Optional[Text] = ...,
        tls: bool = ...,
    ) -> None:
        """Create a new `FTPFS` instance.

        Arguments:
            host (str): A FTP host, e.g. ``'ftp.mirror.nl'``.
            user (str): A username (default is ``'anonymous'``).
            passwd (str): Password for the server, or `None` for anon.
            acct (str): FTP account.
            timeout (int): Timeout for contacting server (in seconds,
                defaults to 10).
            port (int): FTP port number (default 21).
            proxy (str, optional): An FTP proxy, or ``None`` (default)
                for no proxy.
            tls (bool): Attempt to use FTP over TLS (FTPS) (default: False)

        """
        ...
    def __repr__(self) -> Text: ...
    def __str__(self) -> Text: ...
    @property
    def user(self) -> Text: ...
    @property
    def host(self) -> Text: ...
    @property
    def ftp_url(self) -> Text:
        """Get the FTP url this filesystem will open."""
        ...
    @property
    def ftp(self) -> FTP:
        """~ftplib.FTP: the underlying FTP client."""
        ...
    def geturl(self, path: str, purpose: str = ...) -> Text:
        """Get FTP url for resource."""
        ...
    @property
    def features(self) -> Dict[Text, Text]:
        """`dict`: Features of the remote FTP server."""
        ...
    @property
    def supports_mlst(self) -> bool:
        """bool: whether the server supports MLST feature."""
        ...
    def create(self, path: Text, wipe: bool = ...) -> bool: ...
    if typing.TYPE_CHECKING:
        def opendir(
            self: _F, path: Text, factory: Optional[_OpendirFactory] = ...
        ) -> SubFS[_F]: ...
    def getinfo(
        self, path: Text, namespaces: Optional[Container[Text]] = ...
    ) -> Info: ...
    def getmeta(self, namespace: Text = ...) -> Dict[Text, object]: ...
    def listdir(self, path: Text) -> List[Text]: ...
    def makedir(
        self: _F,
        path: Text,
        permissions: Optional[Permissions] = ...,
        recreate: bool = ...,
    ) -> SubFS[_F]: ...
    def openbin(
        self, path: Text, mode: Text = ..., buffering: int = ..., **options: Any
    ) -> BinaryIO: ...
    def remove(self, path: Text) -> None: ...
    def removedir(self, path: Text) -> None: ...
    def scandir(
        self,
        path: Text,
        namespaces: Optional[Container[Text]] = ...,
        page: Optional[Tuple[int, int]] = ...,
    ) -> Iterator[Info]: ...
    def upload(
        self,
        path: Text,
        file: BinaryIO,
        chunk_size: Optional[int] = ...,
        **options: Any
    ) -> None: ...
    def writebytes(self, path: Text, contents: ByteString) -> None: ...
    def setinfo(self, path: Text, info: RawInfo) -> None: ...
    def readbytes(self, path: Text) -> bytes: ...
    def close(self) -> None: ...
