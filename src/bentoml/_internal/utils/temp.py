from __future__ import annotations

import shutil
import tempfile
from collections import deque
from functools import partial
from pathlib import Path
from threading import Lock


class TempfilePool:
    """A simple pool to get temp directories,
    so they are reused as much as possible.
    """

    def __init__(
        self,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,
    ) -> None:
        self._pool: deque[str] = deque([])
        self._lock = Lock()
        self._new = partial(tempfile.mkdtemp, suffix=suffix, prefix=prefix, dir=dir)

    def cleanup(self) -> None:
        while len(self._pool):
            dir = self._pool.popleft()
            shutil.rmtree(dir, ignore_errors=True)

    def acquire(self) -> str:
        with self._lock:
            if not len(self._pool):
                return self._new()
            else:
                return self._pool.popleft()

    def release(self, dir: str) -> None:
        for child in Path(dir).iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        with self._lock:
            self._pool.append(dir)
