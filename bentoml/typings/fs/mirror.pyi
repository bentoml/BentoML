from __future__ import annotations

from typing import Text

from .base import FS
from .walk import Walker

def mirror(
    src_fs: FS | Text,
    dst_fs: FS | Text,
    walker: Walker | None = ...,
    copy_if_newer: bool = ...,
    workers: int = ...,
) -> None: ...
