import typing
from typing import Text, Union
from .base import FS

if typing.TYPE_CHECKING: ...

def move_fs(
    src_fs: Union[Text, FS], dst_fs: Union[Text, FS], workers: int = ...
) -> None: ...
def move_file(
    src_fs: Union[Text, FS], src_path: Text, dst_fs: Union[Text, FS], dst_path: Text
) -> None: ...
def move_dir(
    src_fs: Union[Text, FS],
    src_path: Text,
    dst_fs: Union[Text, FS],
    dst_path: Text,
    workers: int = ...,
) -> None: ...
