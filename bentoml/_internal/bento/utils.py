import os
import threading
import typing as t
from contextlib import contextmanager

_lock: threading.Lock = threading.Lock()


@contextmanager
def change_working_dir(path: t.Optional[str]):
    """Temporarily change current python session's working directory to the given path
    for this context.
    """
    if path is None:
        yield os.getcwd()
        return

    # chdir to build_ctx if exist
    path = os.path.realpath(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"path {path} does not exist")

    global _lock
    with _lock:
        # cwd - current working directory is unique within each Process, this lock make
        # this method thread safe
        oldpwd = os.getcwd()
        os.chdir(path)
        try:
            yield path
        finally:
            os.chdir(oldpwd)


def resolve_user_filepath(filepath: str, ctx: t.Optional[str]) -> str:
    """Resolve the abspath of a filepath provided by user, which may contain "~" or may
    be a relative path base on ctx dir.
    """
    # Return if filepath exist after expanduser
    _path = os.path.expanduser(filepath)
    if os.path.exists(_path):
        return os.path.realpath(_path)

    # Try finding file in ctx if provided
    if ctx:
        _path = os.path.expanduser(os.path.join(ctx, filepath))
        if os.path.exists(_path):
            return os.path.realpath(_path)

    raise FileNotFoundError(f"file {filepath} not found")
