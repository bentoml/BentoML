from __future__ import annotations

import contextlib
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
    dest = os.path.realpath(destination)
    for member in tar.getmembers():
        fn = member.name
        path = os.path.abspath(os.path.join(dest, fn))
        if not Path(path).is_relative_to(dest):
            logger.warning(
                "The tar file has a file (%s) trying to unpack to"
                "outside target directory",
                fn,
            )
            continue
        if member.isdir():
            os.makedirs(path, exist_ok=True)
        elif member.issym():
            target = os.path.normpath(
                os.path.join(os.path.dirname(path), member.linkname)
            )
            if not Path(target).is_relative_to(dest):
                logger.warning(
                    "The tar file has a symlink (%s -> %s) pointing outside"
                    " target directory",
                    fn,
                    member.linkname,
                )
                continue
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
            real_path = os.path.realpath(path)
            if not Path(real_path).is_relative_to(dest):
                logger.warning(
                    "The tar file has a file (%s) resolving outside"
                    " target directory",
                    fn,
                )
                fp.close()
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


def resolve_user_filepath(
    filepath: str, ctx: t.Optional[str], secure: bool = True
) -> str:
    """Resolve the abspath of a filepath provided by user. User provided file path can:
    * be a relative path base on ctx dir
    * contain leading "~" for HOME directory
    * contain environment variables such as "$HOME/workspace"
    """
    # Return if filepath exist after expanduser

    _path = Path(os.path.expanduser(os.path.expandvars(filepath)))

    # Try finding file in ctx if provided
    if not _path.is_absolute():
        ctx = os.path.expanduser(ctx) if ctx else os.getcwd()
        _path = Path(ctx).joinpath(_path)
    elif secure:
        raise ValueError(f"Absolute path {filepath} is not allowed")
    _path = _path.resolve()
    if not _path.exists():
        raise FileNotFoundError(f"file {filepath} not found")
    if secure:
        cwd = Path().resolve()
        if not _path.is_relative_to(cwd):
            raise ValueError(
                f"Accessing file outside of current working directory is not allowed: {_path}"
            )
        if any(part.startswith(".") for part in _path.parts):
            raise ValueError(f"Accessing hidden files is not allowed: {_path}")
        if any(_path.is_relative_to(item) for item in ("/etc", "/proc")):
            raise ValueError(f"Accessing system files is not allowed: {_path}")
    return str(_path)


def safe_remove_dir(path: PathType) -> None:
    with contextlib.suppress(OSError):
        shutil.rmtree(path, ignore_errors=True)


@contextlib.contextmanager
def chdir(new_dir: PathType) -> t.Generator[None, None, None]:
    """Context manager for changing the current working directory. This is not thread-safe."""
    prev_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev_dir)
