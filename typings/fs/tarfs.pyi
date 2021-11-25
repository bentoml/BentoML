

import typing
from typing import Any, BinaryIO, Collection, List, Optional, Text, Tuple, Union

import six

from . import errors
from .base import FS
from .info import Info, RawInfo
from .permissions import Permissions
from .subfs import SubFS
from .wrapfs import WrapFS

"""Manage the filesystem in a Tar archive.
"""
if typing.TYPE_CHECKING:
    T = ...
__all__ = ["TarFS", "WriteTarFS", "ReadTarFS"]
if six.PY2: ...
else: ...

class TarFS(WrapFS):
    """Read and write tar files.

    There are two ways to open a `TarFS` for the use cases of reading
    a tar file, and creating a new one.

    If you open the `TarFS` with  ``write`` set to `False` (the
    default), then the filesystem will be a read only filesystem which
    maps to the files and directories within the tar file. Files are
    decompressed on the fly when you open them.

    Here's how you might extract and print a readme from a tar file::

        with TarFS('foo.tar.gz') as tar_fs:
            readme = tar_fs.readtext('readme.txt')

    If you open the TarFS with ``write`` set to `True`, then the `TarFS`
    will be a empty temporary filesystem. Any files / directories you
    create in the `TarFS` will be written in to a tar file when the `TarFS`
    is closed. The compression is set from the new file name but may be
    set manually with the ``compression`` argument.

    Here's how you might write a new tar file containing a readme.txt
    file::

        with TarFS('foo.tar.xz', write=True) as new_tar:
            new_tar.writetext(
                'readme.txt',
                'This tar file was written by PyFilesystem'
            )

    Arguments:
        file (str or io.IOBase): An OS filename, or an open file handle.
        write (bool): Set to `True` to write a new tar file, or
            use default (`False`) to read an existing tar file.
        compression (str, optional): Compression to use (one of the formats
            supported by `tarfile`: ``xz``, ``gz``, ``bz2``, or `None`).
        temp_fs (str): An FS URL or an FS instance to use to store
            data prior to tarring. Defaults to creating a new
            `~fs.tempfs.TempFS`.

    """

    _compression_formats = ...
    def __new__(
        cls,
        file: Union[Text, BinaryIO],
        write: bool = ...,
        compression: Optional[Text] = ...,
        encoding: Text = ...,
        temp_fs: Union[Text, FS] = ...,
    ) -> FS: ...
    if typing.TYPE_CHECKING:
        def __init__(
            self,
            file: Union[Text, BinaryIO],
            write: bool = ...,
            compression: Optional[Text] = ...,
            encoding: Text = ...,
            temp_fs: Text = ...,
        ) -> None: ...

@six.python_2_unicode_compatible
class WriteTarFS(WrapFS):
    """A writable tar file."""

    def __init__(
        self,
        file: Union[Text, BinaryIO],
        compression: Optional[Text] = ...,
        encoding: Text = ...,
        temp_fs: Union[Text, FS] = ...,
    ) -> None: ...
    def __repr__(self) -> Text: ...
    def __str__(self) -> Text: ...
    def delegate_path(self, path: Text) -> Tuple[FS, Text]: ...
    def delegate_fs(self) -> FS: ...
    def close(self) -> None: ...
    def write_tar(
        self,
        file: Union[Text, BinaryIO, None] = ...,
        compression: Optional[Text] = ...,
        encoding: Optional[Text] = ...,
    ) -> None:
        """Write tar to a file.

        Arguments:
            file (str or io.IOBase, optional): Destination file, may be
                a file name or an open file object.
            compression (str, optional): Compression to use (one of
                the constants defined in `tarfile` in the stdlib).
            encoding (str, optional): The character encoding to use
                (default uses the encoding defined in
                `~WriteTarFS.__init__`).

        Note:
            This is called automatically when the TarFS is closed.

        """
        ...

@six.python_2_unicode_compatible
class ReadTarFS(FS):
    """A readable tar file."""

    _meta = ...
    _typemap = ...
    @errors.CreateFailed.catch_all
    def __init__(self, file: Union[Text, BinaryIO], encoding: Text = ...) -> None: ...
    def __repr__(self) -> Text: ...
    def __str__(self) -> Text: ...
    if six.PY2: ...
    else: ...
    def getinfo(
        self, path: Text, namespaces: Optional[Collection[Text]] = ...
    ) -> Info: ...
    def isdir(self, path): ...
    def isfile(self, path): ...
    def setinfo(self, path: Text, info: RawInfo) -> None: ...
    def listdir(self, path: Text) -> List[Text]: ...
    def makedir(
        self: T,
        path: Text,
        permissions: Optional[Permissions] = ...,
        recreate: bool = ...,
    ) -> SubFS[T]: ...
    def openbin(
        self, path: Text, mode: Text = ..., buffering: int = ..., **options: Any
    ) -> BinaryIO: ...
    def remove(self, path: Text) -> None: ...
    def removedir(self, path: Text) -> None: ...
    def close(self) -> None: ...
    def isclosed(self) -> bool: ...
    def geturl(self, path: Text, purpose: Text = ...) -> Text: ...

if __name__ == "__main__": ...
