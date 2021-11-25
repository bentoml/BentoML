

import typing
from typing import Text, Union

from .base import FS

"""Functions for moving files between filesystems.
"""
if typing.TYPE_CHECKING: ...

def move_fs(
    src_fs: Union[Text, FS], dst_fs: Union[Text, FS], workers: int = ...
) -> None:
    """Move the contents of a filesystem to another filesystem.

    Arguments:
        src_fs (FS or str): Source filesystem (instance or URL).
        dst_fs (FS or str): Destination filesystem (instance or URL).
        workers (int): Use `worker` threads to copy data, or ``0`` (default) for
            a single-threaded copy.

    """
    ...

def move_file(
    src_fs: Union[Text, FS], src_path: Text, dst_fs: Union[Text, FS], dst_path: Text
) -> None:
    """Move a file from one filesystem to another.

    Arguments:
        src_fs (FS or str): Source filesystem (instance or URL).
        src_path (str): Path to a file on ``src_fs``.
        dst_fs (FS or str): Destination filesystem (instance or URL).
        dst_path (str): Path to a file on ``dst_fs``.

    """
    ...

def move_dir(
    src_fs: Union[Text, FS],
    src_path: Text,
    dst_fs: Union[Text, FS],
    dst_path: Text,
    workers: int = ...,
) -> None:
    """Move a directory from one filesystem to another.

    Arguments:
        src_fs (FS or str): Source filesystem (instance or URL).
        src_path (str): Path to a directory on ``src_fs``
        dst_fs (FS or str): Destination filesystem (instance or URL).
        dst_path (str): Path to a directory on ``dst_fs``.
        workers (int): Use ``worker`` threads to copy data, or ``0``
            (default) for a single-threaded copy.

    """
    ...
