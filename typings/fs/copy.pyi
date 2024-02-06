from typing import Callable
from typing import Optional
from typing import Text
from typing import Union

from .base import FS
from .walk import Walker

_OnCopy = Callable[[FS, Text, FS, Text], object]

def copy_fs(
    src_fs: Union[FS, Text],
    dst_fs: Union[FS, Text],
    walker: Optional[Walker] = ...,
    on_copy: Optional[_OnCopy] = ...,
    workers: int = ...,
) -> None: ...
def copy_fs_if_newer(
    src_fs: Union[FS, Text],
    dst_fs: Union[FS, Text],
    walker: Optional[Walker] = ...,
    on_copy: Optional[_OnCopy] = ...,
    workers: int = ...,
) -> None: ...
def copy_file(
    src_fs: Union[FS, Text], src_path: Text, dst_fs: Union[FS, Text], dst_path: Text
) -> None: ...
def copy_file_internal(
    src_fs: FS, src_path: Text, dst_fs: FS, dst_path: Text
) -> None: ...
def copy_file_if_newer(
    src_fs: Union[FS, Text], src_path: Text, dst_fs: Union[FS, Text], dst_path: Text
) -> bool: ...
def copy_structure(
    src_fs: Union[FS, Text], dst_fs: Union[FS, Text], walker: Optional[Walker] = ...
) -> None: ...
def copy_dir(
    src_fs: Union[FS, Text],
    src_path: Text,
    dst_fs: Union[FS, Text],
    dst_path: Text,
    walker: Optional[Walker] = ...,
    on_copy: Optional[_OnCopy] = ...,
    workers: int = ...,
) -> None: ...
def copy_dir_if_newer(
    src_fs: Union[FS, Text],
    src_path: Text,
    dst_fs: Union[FS, Text],
    dst_path: Text,
    walker: Optional[Walker] = ...,
    on_copy: Optional[_OnCopy] = ...,
    workers: int = ...,
) -> None: ...
