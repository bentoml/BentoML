

import typing
from typing import BinaryIO, Optional, Text, Tuple, Union

from .base import FS
from .walk import Walker

"""Functions to compress the contents of a filesystem.

Currently zip and tar are supported, using the `zipfile` and
`tarfile` modules from the standard library.
"""
if typing.TYPE_CHECKING:
    ZipTime = Tuple[int, int, int, int, int, int]

def write_zip(
    src_fs: FS,
    file: Union[Text, BinaryIO],
    compression: int = ...,
    encoding: Text = ...,
    walker: Optional[Walker] = ...,
) -> None:
    """Write the contents of a filesystem to a zip file.

    Arguments:
        src_fs (~fs.base.FS): The source filesystem to compress.
        file (str or io.IOBase): Destination file, may be a file name
            or an open file object.
        compression (int): Compression to use (one of the constants
            defined in the `zipfile` module in the stdlib). Defaults
            to `zipfile.ZIP_DEFLATED`.
        encoding (str): The encoding to use for filenames. The default
            is ``"utf-8"``, use ``"CP437"`` if compatibility with WinZip
            is desired.
        walker (~fs.walk.Walker, optional): A `Walker` instance, or `None`
            to use default walker. You can use this to specify which files
            you want to compress.

    """
    ...

def write_tar(
    src_fs: FS,
    file: Union[Text, BinaryIO],
    compression: Optional[Text] = ...,
    encoding: Text = ...,
    walker: Optional[Walker] = ...,
) -> None:
    """Write the contents of a filesystem to a tar file.

    Arguments:
        src_fs (~fs.base.FS): The source filesystem to compress.
        file (str or io.IOBase): Destination file, may be a file
            name or an open file object.
        compression (str, optional): Compression to use, or `None`
            for a plain Tar archive without compression.
        encoding(str): The encoding to use for filenames. The
            default is ``"utf-8"``.
        walker (~fs.walk.Walker, optional): A `Walker` instance, or
            `None` to use default walker. You can use this to specify
            which files you want to compress.

    """
    ...
