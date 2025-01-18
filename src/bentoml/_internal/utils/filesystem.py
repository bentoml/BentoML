from __future__ import annotations

import logging
import os
import shutil
import tarfile
import tempfile
import typing as t
from collections import deque
from functools import partial
from pathlib import Path
from threading import Lock

import fs
import fs.copy
from fs.base import FS

from bentoml._internal.utils.uri import encode_path_for_uri

if t.TYPE_CHECKING:
    from ..types import PathType

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


def calc_dir_size(path: PathType) -> int:
    return sum(f.stat().st_size for f in Path(path).glob("**/*") if f.is_file())


def validate_or_create_dir(*path: PathType) -> None:
    for p in path:
        path_obj = Path(p)

        if path_obj.exists():
            if not path_obj.is_dir():
                raise OSError(20, f"{path_obj} is not a directory")
        else:
            path_obj.mkdir(parents=True, exist_ok=True)


def copy_file_to_fs_folder(
    src_path: str,
    dst_fs: FS,
    dst_folder_path: str = ".",
    dst_filename: t.Optional[str] = None,
):
    """Copy the given file at src_path to dst_fs filesystem, under its dst_folder_path
    folder with dst_filename as file name. When dst_filename is None, keep the original
    file name.
    """
    src_path = os.path.realpath(os.path.expanduser(src_path))
    dir_name, file_name = os.path.split(src_path)
    src_fs = fs.open_fs(encode_path_for_uri(dir_name))
    dst_filename = file_name if dst_filename is None else dst_filename
    dst_path = fs.path.join(dst_folder_path, dst_filename)
    dst_fs.makedir(dst_folder_path, recreate=True)
    fs.copy.copy_file(src_fs, file_name, dst_fs, dst_path)


def resolve_user_filepath(filepath: str, ctx: t.Optional[str]) -> str:
    """Resolve the abspath of a filepath provided by user. User provided file path can:
    * be a relative path base on ctx dir
    * contain leading "~" for HOME directory
    * contain environment variables such as "$HOME/workspace"
    """
    # Return if filepath exist after expanduser

    _path = os.path.expanduser(os.path.expandvars(filepath))

    # Try finding file in ctx if provided
    if not os.path.isabs(_path) and ctx:
        _path = os.path.expanduser(os.path.join(ctx, filepath))

    if os.path.exists(_path):
        return os.path.realpath(_path)

    raise FileNotFoundError(f"file {filepath} not found")
