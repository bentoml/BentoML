from __future__ import annotations

import os
import sys
import uuid
import random
import socket
import typing as t
import functools
import contextlib
from typing import overload
from typing import TYPE_CHECKING
from pathlib import Path
from datetime import date
from datetime import time
from datetime import datetime
from datetime import timedelta

import fs
import attr
import fs.copy

if sys.version_info >= (3, 8):
    from functools import cached_property
else:
    from backports.cached_property import cached_property

from .cattr import bentoml_cattr
from ..types import LazyType
from ..types import PathType
from ..types import MetadataDict
from ..types import MetadataType
from .lazy_loader import LazyLoader

if TYPE_CHECKING:
    from fs.base import FS

    P = t.ParamSpec("P")
    GenericFunction = t.Callable[P, t.Any]


C = t.TypeVar("C")
T = t.TypeVar("T")
_T_co = t.TypeVar("_T_co", covariant=True, bound=t.Any)


__all__ = [
    "bentoml_cattr",
    "cached_property",
    "cached_contextmanager",
    "reserve_free_port",
    "catch_exceptions",
    "LazyLoader",
    "validate_or_create_dir",
    "display_path_under_home",
]


@overload
def kwargs_transformers(
    func: GenericFunction[t.Concatenate[str, bool, t.Iterable[str], P]],
    *,
    transformer: GenericFunction[t.Any],
) -> GenericFunction[t.Concatenate[str, t.Iterable[str], bool, P]]:
    ...


@overload
def kwargs_transformers(
    func: None = None, *, transformer: GenericFunction[t.Any]
) -> GenericFunction[t.Any]:
    ...


def kwargs_transformers(
    _func: t.Callable[..., t.Any] | None = None,
    *,
    transformer: GenericFunction[t.Any],
) -> GenericFunction[t.Any]:
    def decorator(func: GenericFunction[t.Any]) -> t.Callable[P, t.Any]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> t.Any:
            return func(*args, **{k: transformer(v) for k, v in kwargs.items()})

        return wrapper

    if _func is None:
        return decorator
    return decorator(_func)


@t.overload
def first_not_none(*args: T | None, default: T) -> T:
    ...


@t.overload
def first_not_none(*args: T | None) -> T | None:
    ...


def first_not_none(*args: T | None, default: None | T = None) -> T | None:
    """
    Returns the first argument that is not None.
    """
    return next((arg for arg in args if arg is not None), default)


def randomize_runner_name(module_name: str):
    return f"{module_name.split('.')[-1]}_{uuid.uuid4().hex[:6].lower()}"


def validate_or_create_dir(*path: PathType) -> None:
    for p in path:
        path_obj = Path(p)

        if path_obj.exists():
            if not path_obj.is_dir():
                raise OSError(20, f"{path_obj} is not a directory")
        else:
            path_obj.mkdir(parents=True)


def calc_dir_size(path: PathType) -> int:
    return sum(f.stat().st_size for f in Path(path).glob("**/*") if f.is_file())


def display_path_under_home(path: str) -> str:
    # Shorten path under home directory with leading `~`
    # e.g. from `/Users/foo/bar` to just `~/bar`
    try:
        return str("~" / Path(path).relative_to(Path.home()))
    except ValueError:
        # when path is not under home directory, return original full path
        return path


def human_readable_size(size: t.Union[int, float], decimal_places: int = 2) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if size < 1024.0 or unit == "PiB":
            break
        size /= 1024.0
    else:
        raise ValueError("size is too large")
    return f"{size:.{decimal_places}f} {unit}"


class catch_exceptions(t.Generic[_T_co], object):
    def __init__(
        self,
        catch_exc: t.Union[t.Type[BaseException], t.Tuple[t.Type[BaseException], ...]],
        throw_exc: t.Callable[[str], BaseException],
        msg: str = "",
        fallback: t.Optional[_T_co] = None,
        raises: t.Optional[bool] = True,
    ) -> None:
        self._catch_exc = catch_exc
        self._throw_exc = throw_exc
        self._msg = msg
        self._fallback = fallback
        self._raises = raises

    def __call__(self, func: t.Callable[P, _T_co]) -> t.Callable[P, t.Optional[_T_co]]:
        @functools.wraps(func)
        def _(*args: P.args, **kwargs: P.kwargs) -> t.Optional[_T_co]:
            try:
                return func(*args, **kwargs)
            except self._catch_exc:
                if self._raises:
                    raise self._throw_exc(self._msg)
                return self._fallback

        return _


@contextlib.contextmanager
def reserve_free_port(
    host: str = "localhost",
    prefix: t.Optional[str] = None,
    max_retry: int = 50,
) -> t.Iterator[int]:
    """
    detect free port and reserve until exit the context
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if prefix is not None:
        prefix_num = int(prefix) * 10 ** (5 - len(prefix))
        suffix_range = min(65535 - prefix_num, 10 ** (5 - len(prefix)))
        for _ in range(max_retry):
            suffix = random.randint(0, suffix_range)
            port = int(f"{prefix_num + suffix}")
            try:
                sock.bind((host, port))
                break
            except OSError:
                continue
        else:
            raise RuntimeError(
                f"cannot find free port with prefix {prefix} after {max_retry} retries"
            )
    else:
        sock.bind((host, 0))
    port = sock.getsockname()[1]
    yield port
    sock.close()


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
    src_fs = fs.open_fs(dir_name)
    dst_filename = file_name if dst_filename is None else dst_filename
    dst_path = fs.path.join(dst_folder_path, dst_filename)
    fs.copy.copy_file(src_fs, file_name, dst_fs, dst_path)


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


def label_validator(
    _: t.Any, _attr: attr.Attribute[dict[str, str]], labels: dict[str, str]
):
    validate_labels(labels)


def validate_labels(labels: dict[str, str]):
    if not isinstance(labels, dict):
        raise ValueError("labels must be a dict!")

    for key, val in labels.items():
        if not isinstance(key, str):
            raise ValueError("label keys must be strings")

        if not isinstance(val, str):
            raise ValueError("label values must be strings")


def metadata_validator(
    _: t.Any, _attr: attr.Attribute[MetadataDict], metadata: MetadataDict
):
    validate_metadata(metadata)


def validate_metadata(metadata: MetadataDict):
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dict!")

    for key, val in metadata.items():
        if not isinstance(key, (str, int, float)):
            raise ValueError("metadata keys must be strings")

        metadata[key] = _validate_metadata_entry(val)


def _validate_metadata_entry(entry: MetadataType) -> MetadataType:
    if isinstance(entry, dict):
        validate_metadata(entry)
    elif isinstance(entry, list):
        for i, val in enumerate(entry):
            entry[i] = _validate_metadata_entry(val)
    elif isinstance(entry, tuple):
        entry = tuple((_validate_metadata_entry(x) for x in entry))

    elif LazyType("numpy", "ndarray").isinstance(entry):
        entry = entry.tolist()  # type: ignore (LazyType)
        _validate_metadata_entry(entry)
    elif LazyType("numpy", "generic").isinstance(entry):
        entry = entry.item()  # type: ignore (LazyType)
        _validate_metadata_entry(entry)
    elif LazyType("scipy.sparse", "spmatrix").isinstance(entry):
        raise ValueError(
            "SciPy sparse matrices are currently not supported as metadata items; consider using a dictionary instead"
        )
    elif LazyType("pandas", "Series").isinstance(entry):
        entry = {entry.name: entry.to_dict()}
        _validate_metadata_entry(entry)
    elif LazyType("pandas.api.extensions", "ExtensionArray").isinstance(entry):
        entry = entry.to_numpy().tolist()  # type: ignore (LazyType)
        _validate_metadata_entry(entry)
    elif LazyType("pandas", "DataFrame").isinstance(entry):
        entry = entry.to_dict()  # type: ignore (LazyType)
        validate_metadata(entry)  # type: ignore (LazyType)
    elif LazyType("pandas", "Timestamp").isinstance(entry):
        entry = entry.to_pydatetime()  # type: ignore (LazyType)
    elif LazyType("pandas", "Timedelta").isinstance(entry):
        entry = entry.to_pytimedelta()  # type: ignore (LazyType)
    elif LazyType("pandas", "Period").isinstance(entry):
        entry = entry.to_timestamp().to_pydatetime()  # type: ignore (LazyType)
    elif LazyType("pandas", "Interval").isinstance(entry):
        entry = (entry.left, entry.right)  # type: ignore (LazyType)
        _validate_metadata_entry(entry)
    elif not isinstance(
        entry, (str, bytes, bool, int, float, complex, datetime, date, time, timedelta)
    ):
        raise ValueError(
            f"metadata entries must be basic python types like 'str', 'int', or 'complex', got '{type(entry).__name__}'"
        )

    return entry


VT = t.TypeVar("VT")


class cached_contextmanager:
    """
    Just like contextlib.contextmanager, but will cache the yield value for the same
    arguments. When all instances of the same contextmanager exits, the cache value will
    be dropped.
    Example Usage:
    (To reuse the container based on the same image)
    >>> @cached_contextmanager("{docker_image.id}")
    >>> def start_docker_container_from_image(docker_image, timeout=60):
    >>>     container = ...
    >>>     yield container
    >>>     container.stop()
    """

    def __init__(self, cache_key_template: t.Optional[str] = None):
        self._cache_key_template = cache_key_template
        self._cache: t.Dict[t.Any, t.Any] = {}

    def __call__(
        self, func: "t.Callable[P, t.Generator[VT, None, None]]"
    ) -> "t.Callable[P, t.ContextManager[VT]]":

        func_m = contextlib.contextmanager(func)

        @contextlib.contextmanager
        @functools.wraps(func)
        def _func(*args: "P.args", **kwargs: "P.kwargs") -> t.Any:
            import inspect

            bound_args = inspect.signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()
            if self._cache_key_template:
                cache_key = self._cache_key_template.format(**bound_args.arguments)
            else:
                cache_key = repr(tuple(bound_args.arguments.values()))
            if cache_key in self._cache:
                yield self._cache[cache_key]
            else:
                with func_m(*args, **kwargs) as value:
                    self._cache[cache_key] = value
                    yield value
                    self._cache.pop(cache_key)

        return _func
