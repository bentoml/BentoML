import typing
from typing import BinaryIO, Optional, Text, Tuple, Union
from .base import FS
from .walk import Walker

if typing.TYPE_CHECKING:
    ZipTime = Tuple[int, int, int, int, int, int]

def write_zip(
    src_fs: FS,
    file: Union[Text, BinaryIO],
    compression: int = ...,
    encoding: Text = ...,
    walker: Optional[Walker] = ...,
) -> None: ...
def write_tar(
    src_fs: FS,
    file: Union[Text, BinaryIO],
    compression: Optional[Text] = ...,
    encoding: Text = ...,
    walker: Optional[Walker] = ...,
) -> None: ...
