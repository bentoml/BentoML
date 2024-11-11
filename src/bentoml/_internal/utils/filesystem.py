from __future__ import annotations

import logging
import os
import shutil
import tarfile
import tempfile
from collections import deque
from functools import partial
from pathlib import Path
from threading import Lock

import fs

logger = logging.getLogger(__name__)


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


def safe_extract_tarfile(tar: tarfile.TarFile, destination: str) -> None:
    # Borrowed from pip but continue on error
    os.makedirs(destination, exist_ok=True)
    for member in tar.getmembers():
        fn = member.name
        path = os.path.join(destination, fn)
        if not fs.path.relativefrom(destination, path):
            logger.warning(
                "The tar file has a file (%s) trying to unpack to"
                "outside target directory",
                fn,
            )
            continue
        if member.isdir():
            os.makedirs(path, exist_ok=True)
        elif member.issym():
            try:
                tar._extract_member(member, path)
            except Exception as exc:
                # Some corrupt tar files seem to produce this
                # (specifically bad symlinks)
                logger.warning("In the tar file the member %s is invalid: %s", fn, exc)
                continue
        else:
            try:
                fp = tar.extractfile(member)
            except (KeyError, AttributeError) as exc:
                # Some corrupt tar files seem to produce this
                # (specifically bad symlinks)
                logger.warning("In the tar file the member %s is invalid: %s", fn, exc)
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if fp is None:
                continue
            with open(path, "wb") as destfp:
                shutil.copyfileobj(fp, destfp)
            fp.close()
            # Update the timestamp (useful for cython compiled files)
            tar.utime(member, path)
