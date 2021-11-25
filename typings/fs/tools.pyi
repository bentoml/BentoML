

import typing
from typing import IO, List, Optional, Text

from .base import FS

"""Miscellaneous tools for operating on filesystems.
"""
if typing.TYPE_CHECKING: ...

def remove_empty(fs: FS, path: Text) -> None:
    """Remove all empty parents.

    Arguments:
        fs (FS): A filesystem instance.
        path (str): Path to a directory on the filesystem.

    """
    ...

def copy_file_data(src_file: IO, dst_file: IO, chunk_size: Optional[int] = ...) -> None:
    """Copy data from one file object to another.

    Arguments:
        src_file (io.IOBase): File open for reading.
        dst_file (io.IOBase): File open for writing.
        chunk_size (int): Number of bytes to copy at
            a time (or `None` to use sensible default).

    """
    ...

def get_intermediate_dirs(fs: FS, dir_path: Text) -> List[Text]:
    """Get a list of non-existing intermediate directories.

    Arguments:
        fs (FS): A filesystem instance.
        dir_path (str): A path to a new directory on the filesystem.

    Returns:
        list: A list of non-existing paths.

    Raises:
        ~fs.errors.DirectoryExpected: If a path component
            references a file and not a directory.

    """
    ...

def is_thread_safe(*filesystems: FS) -> bool:
    """Check if all filesystems are thread-safe.

    Arguments:
        filesystems (FS): Filesystems instances to check.

    Returns:
        bool: if all filesystems are thread safe.

    """
    ...
