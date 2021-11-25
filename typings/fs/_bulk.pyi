

import threading
import typing
from types import TracebackType
from typing import IO, Optional, Text, Type

from .base import FS

"""

Implements a thread pool for parallel copying of files.

"""
if typing.TYPE_CHECKING: ...

class _Worker(threading.Thread):
    """Worker thread that pulls tasks from a queue."""

    def __init__(self, copier) -> None: ...
    def run(self): ...

class _Task:
    """Base class for a task."""

    def __call__(self) -> None:
        """Task implementation."""
        ...

class _CopyTask(_Task):
    """A callable that copies from one file another."""

    def __init__(self, src_file: IO, dst_file: IO) -> None: ...
    def __call__(self) -> None: ...

class Copier:
    """Copy files in worker threads."""

    def __init__(self, num_workers: int = ...) -> None: ...
    def start(self):  # -> None:
        """Start the workers."""
        ...
    def stop(self):  # -> None:
        """Stop the workers (will block until they are finished)."""
        ...
    def add_error(self, error):  # -> None:
        """Add an exception raised by a task."""
        ...
    def __enter__(self): ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ): ...
    def copy(self, src_fs: FS, src_path: Text, dst_fs: FS, dst_path: Text) -> None:
        """Copy a file from one fs to another."""
        ...
