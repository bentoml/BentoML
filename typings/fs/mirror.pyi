import typing
from typing import Optional, Text, Union
from .base import FS
from .walk import Walker

if typing.TYPE_CHECKING: ...

def mirror(
    src_fs: Union[FS, Text],
    dst_fs: Union[FS, Text],
    walker: Optional[Walker] = ...,
    copy_if_newer: bool = ...,
    workers: int = ...,
) -> None: ...
