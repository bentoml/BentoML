import dataclasses
import zipfile
from collections import abc
from io import BytesIO
from typing import IO, Any, AnyStr

from pandas._typing import (
    Buffer,
    CompressionDict,
    CompressionOptions,
    FileOrBuffer,
    FilePathOrBuffer,
    StorageOptions,
)

"""Common IO api utilities"""
lzma = ...
_VALID_URLS = ...

@dataclasses.dataclass
class IOArgs:
    """
    Return value of io/common.py:_get_filepath_or_buffer.

    Note (copy&past from io/parsers):
    filepath_or_buffer can be Union[FilePathOrBuffer, s3fs.S3File, gcsfs.GCSFile]
    though mypy handling of conditional imports is difficult.
    See https://github.com/python/mypy/issues/1297
    """

    filepath_or_buffer: FileOrBuffer
    encoding: str
    mode: str
    compression: CompressionDict
    should_close: bool = ...

@dataclasses.dataclass
class IOHandles:
    """
    Return value of io/common.py:get_handle

    Can be used as a context manager.

    This is used to easily close created buffers and to handle corner cases when
    TextIOWrapper is inserted.

    handle: The file handle to be used.
    created_handles: All file handles that are created by get_handle
    is_wrapped: Whether a TextIOWrapper needs to be detached.
    """

    handle: Buffer
    compression: CompressionDict
    created_handles: list[Buffer] = ...
    is_wrapped: bool = ...
    is_mmap: bool = ...
    def close(self) -> None:
        """
        Close all created buffers.

        Note: If a TextIOWrapper was inserted, it is flushed and detached to
        avoid closing the potentially user-created buffer.
        """
        ...
    def __enter__(self) -> IOHandles: ...
    def __exit__(self, *args: Any) -> None: ...

def is_url(url) -> bool:
    """
    Check to see if a URL has a valid protocol.

    Parameters
    ----------
    url : str or unicode

    Returns
    -------
    isurl : bool
        If `url` has a valid protocol return True otherwise False.
    """
    ...

def validate_header_arg(header) -> None: ...
def stringify_path(
    filepath_or_buffer: FilePathOrBuffer[AnyStr], convert_file_like: bool = ...
) -> FileOrBuffer[AnyStr]:
    """
    Attempt to convert a path-like object to a string.

    Parameters
    ----------
    filepath_or_buffer : object to be converted

    Returns
    -------
    str_filepath_or_buffer : maybe a string version of the object

    Notes
    -----
    Objects supporting the fspath protocol (python 3.6+) are coerced
    according to its __fspath__ method.

    Any other object is passed through unchanged, which includes bytes,
    strings, buffers, or anything else that's not even path-like.
    """
    ...

def urlopen(*args, **kwargs):  # -> _UrlopenRet:
    """
    Lazy-import wrapper for stdlib urlopen, as that imports a big chunk of
    the stdlib.
    """
    ...

def is_fsspec_url(url: FilePathOrBuffer) -> bool:
    """
    Returns true if the given URL looks like
    something fsspec can handle
    """
    ...

def file_path_to_url(path: str) -> str:
    """
    converts an absolute native path to a FILE URL.

    Parameters
    ----------
    path : a path in native format

    Returns
    -------
    a valid FILE URL
    """
    ...

_compression_to_extension = ...

def get_compression_method(
    compression: CompressionOptions,
) -> tuple[str | None, CompressionDict]:
    """
    Simplifies a compression argument to a compression method string and
    a mapping containing additional arguments.

    Parameters
    ----------
    compression : str or mapping
        If string, specifies the compression method. If mapping, value at key
        'method' specifies compression method.

    Returns
    -------
    tuple of ({compression method}, Optional[str]
              {compression arguments}, Dict[str, Any])

    Raises
    ------
    ValueError on mapping missing 'method' key
    """
    ...

def infer_compression(
    filepath_or_buffer: FilePathOrBuffer, compression: str | None
) -> str | None:
    """
    Get the compression method for filepath_or_buffer. If compression='infer',
    the inferred compression method is returned. Otherwise, the input
    compression method is returned unchanged, unless it's invalid, in which
    case an error is raised.

    Parameters
    ----------
    filepath_or_buffer : str or file handle
        File path or object.
    compression : {'infer', 'gzip', 'bz2', 'zip', 'xz', None}
        If 'infer' and `filepath_or_buffer` is path-like, then detect
        compression from the following extensions: '.gz', '.bz2', '.zip',
        or '.xz' (otherwise no compression).

    Returns
    -------
    string or None

    Raises
    ------
    ValueError on invalid compression specified.
    """
    ...

def get_handle(
    path_or_buf: FilePathOrBuffer,
    mode: str,
    encoding: str | None = ...,
    compression: CompressionOptions = ...,
    memory_map: bool = ...,
    is_text: bool = ...,
    errors: str | None = ...,
    storage_options: StorageOptions = ...,
) -> IOHandles:
    """
    Get file handle for given path/buffer and mode.

    Parameters
    ----------
    path_or_buf : str or file handle
        File path or object.
    mode : str
        Mode to open path_or_buf with.
    encoding : str or None
        Encoding to use.
    compression : str or dict, default None
        If string, specifies compression mode. If dict, value at key 'method'
        specifies compression mode. Compression mode must be one of {'infer',
        'gzip', 'bz2', 'zip', 'xz', None}. If compression mode is 'infer'
        and `filepath_or_buffer` is path-like, then detect compression from
        the following extensions: '.gz', '.bz2', '.zip', or '.xz' (otherwise
        no compression). If dict and compression mode is one of
        {'zip', 'gzip', 'bz2'}, or inferred as one of the above,
        other entries passed as additional compression options.

        .. versionchanged:: 1.0.0

           May now be a dict with key 'method' as compression mode
           and other keys as compression options if compression
           mode is 'zip'.

        .. versionchanged:: 1.1.0

           Passing compression options as keys in dict is now
           supported for compression modes 'gzip' and 'bz2' as well as 'zip'.

    memory_map : bool, default False
        See parsers._parser_params for more information.
    is_text : bool, default True
        Whether the type of the content passed to the file/buffer is string or
        bytes. This is not the same as `"b" not in mode`. If a string content is
        passed to a binary file/buffer, a wrapper is inserted.
    errors : str, default 'strict'
        Specifies how encoding and decoding errors are to be handled.
        See the errors argument for :func:`open` for a full list
        of options.
    storage_options: StorageOptions = None
        Passed to _get_filepath_or_buffer

    .. versionchanged:: 1.2.0

    Returns the dataclass IOHandles
    """
    ...

class _BytesZipFile(zipfile.ZipFile, BytesIO):
    """
    Wrapper for standard library class ZipFile and allow the returned file-like
    handle to accept byte strings via `write` method.

    BytesIO provides attributes of file-like object and ZipFile.writestr writes
    bytes strings into a member of the archive.
    """

    def __init__(
        self, file: FilePathOrBuffer, mode: str, archive_name: str | None = ..., **kwargs
    ) -> None: ...
    def write(self, data): ...
    def flush(self) -> None: ...
    def close(self): ...
    @property
    def closed(self): ...

class _MMapWrapper(abc.Iterator):
    """
    Wrapper for the Python's mmap class so that it can be properly read in
    by Python's csv.reader class.

    Parameters
    ----------
    f : file object
        File object to be mapped onto memory. Must support the 'fileno'
        method or have an equivalent attribute

    """

    def __init__(
        self, f: IO, encoding: str = ..., errors: str = ..., decode: bool = ...
    ) -> None: ...
    def __getattr__(self, name: str): ...
    def __iter__(self) -> _MMapWrapper: ...
    def read(self, size: int = ...) -> str | bytes: ...
    def __next__(self) -> str: ...

def file_exists(filepath_or_buffer: FilePathOrBuffer) -> bool:
    """Test whether file exists."""
    ...
