

import typing
from typing import (
    Any,
    BinaryIO,
    Collection,
    List,
    Optional,
    SupportsInt,
    Text,
    Tuple,
    Union,
)

import six

from . import errors
from .base import FS
from .info import Info, RawInfo
from .iotools import RawWrapper
from .permissions import Permissions
from .subfs import SubFS
from .wrapfs import WrapFS

"""Manage the filesystem in a Zip archive.
"""
if typing.TYPE_CHECKING:
    R = ...

class _ZipExtFile(RawWrapper):
    def __init__(self, fs: ReadZipFS, name: Text) -> None: ...
    def read(self, size: int = ...) -> bytes: ...
    def read1(self, size: int = ...) -> bytes: ...
    def seek(self, offset: int, whence: SupportsInt = ...) -> int:
        """Change stream position.

        Change the stream position to the given byte offset. The
        offset is interpreted relative to the position indicated by
        ``whence``.

        Arguments:
            offset (int): the offset to the new position, in bytes.
            whence (int): the position reference. Possible values are:
                * `Seek.set`: start of stream (the default).
                * `Seek.current`: current position; offset may be negative.
                * `Seek.end`: end of stream; offset must be negative.

        Returns:
            int: the new absolute position.

        Raises:
            ValueError: when ``whence`` is not known, or ``offset``
                is invalid.

        Note:
            Zip compression does not support seeking, so the seeking
            is emulated. Seeking somewhere else than the current position
            will need to either:
                * reopen the file and restart decompression
                * read and discard data to advance in the file

        """
        ...
    def tell(self) -> int: ...

class ZipFS(WrapFS):
    """Read and write zip files.

    There are two ways to open a `ZipFS` for the use cases of reading
    a zip file, and creating a new one.

    If you open the `ZipFS` with  ``write`` set to `False` (the default)
    then the filesystem will be a read-only filesystem which maps to
    the files and directories within the zip file. Files are
    decompressed on the fly when you open them.

    Here's how you might extract and print a readme from a zip file::

        with ZipFS('foo.zip') as zip_fs:
            readme = zip_fs.readtext('readme.txt')

    If you open the `ZipFS` with ``write`` set to `True`, then the `ZipFS`
    will be an empty temporary filesystem. Any files / directories you
    create in the `ZipFS` will be written in to a zip file when the `ZipFS`
    is closed.

    Here's how you might write a new zip file containing a ``readme.txt``
    file::

        with ZipFS('foo.zip', write=True) as new_zip:
            new_zip.writetext(
                'readme.txt',
                'This zip file was written by PyFilesystem'
            )


    Arguments:
        file (str or io.IOBase): An OS filename, or an open file object.
        write (bool): Set to `True` to write a new zip file, or `False`
            (default) to read an existing zip file.
        compression (int): Compression to use (one of the constants
            defined in the `zipfile` module in the stdlib).
        temp_fs (str or FS): An FS URL or an FS instance to use to
            store data prior to zipping. Defaults to creating a new
            `~fs.tempfs.TempFS`.

    """

    def __new__(
        cls,
        file: Union[Text, BinaryIO],
        write: bool = ...,
        compression: int = ...,
        encoding: Text = ...,
        temp_fs: Union[Text, FS] = ...,
    ) -> FS: ...
    if typing.TYPE_CHECKING:
        def __init__(
            self,
            file: Union[Text, BinaryIO],
            write: bool = ...,
            compression: int = ...,
            encoding: Text = ...,
            temp_fs: Text = ...,
        ) -> None: ...

@six.python_2_unicode_compatible
class WriteZipFS(WrapFS):
    """A writable zip file."""

    def __init__(
        self,
        file: Union[Text, BinaryIO],
        compression: int = ...,
        encoding: Text = ...,
        temp_fs: Union[Text, FS] = ...,
    ) -> None: ...
    def __repr__(self) -> Text: ...
    def __str__(self) -> Text: ...
    def delegate_path(self, path: Text) -> Tuple[FS, Text]: ...
    def delegate_fs(self) -> FS: ...
    def close(self) -> None: ...
    def write_zip(
        self,
        file: Union[Text, BinaryIO, None] = ...,
        compression: Optional[int] = ...,
        encoding: Optional[Text] = ...,
    ) -> None:
        """Write zip to a file.

        Arguments:
            file (str or io.IOBase, optional): Destination file, may be
                a file name or an open file handle.
            compression (int, optional): Compression to use (one of the
                constants defined in the `zipfile` module in the stdlib).
            encoding (str, optional): The character encoding to use
                (default uses the encoding defined in
                `~WriteZipFS.__init__`).

        Note:
            This is called automatically when the ZipFS is closed.

        """
        ...

@six.python_2_unicode_compatible
class ReadZipFS(FS):
    """A readable zip file."""

    _meta = ...
    @errors.CreateFailed.catch_all
    def __init__(self, file: Union[BinaryIO, Text], encoding: Text = ...) -> None: ...
    def __repr__(self) -> Text: ...
    def __str__(self) -> Text: ...
    def getinfo(
        self, path: Text, namespaces: Optional[Collection[Text]] = ...
    ) -> Info: ...
    def setinfo(self, path: Text, info: RawInfo) -> None: ...
    def listdir(self, path: Text) -> List[Text]: ...
    def makedir(
        self: R,
        path: Text,
        permissions: Optional[Permissions] = ...,
        recreate: bool = ...,
    ) -> SubFS[R]: ...
    def openbin(
        self, path: Text, mode: Text = ..., buffering: int = ..., **kwargs: Any
    ) -> BinaryIO: ...
    def remove(self, path: Text) -> None: ...
    def removedir(self, path: Text) -> None: ...
    def close(self) -> None: ...
    def readbytes(self, path: Text) -> bytes: ...
    def geturl(self, path: Text, purpose: Text = ...) -> Text: ...
