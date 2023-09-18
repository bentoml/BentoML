from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import logging
import os
import random
import re
import socket
import sys
import typing as t
from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from pathlib import Path
from reprlib import recursive_repr as _recursive_repr
from typing import TYPE_CHECKING
from typing import overload

import attr
import fs
import fs.copy
from rich.console import Console

if sys.version_info >= (3, 8):
    from functools import cached_property
else:
    from backports.cached_property import cached_property

from ..types import LazyType
from .cattr import bentoml_cattr
from .lazy_loader import LazyLoader

if TYPE_CHECKING:
    from fs.base import FS
    from typing_extensions import Self

    from ..types import MetadataDict
    from ..types import MetadataType
    from ..types import PathType

    P = t.ParamSpec("P")
    F = t.Callable[P, t.Any]


C = t.TypeVar("C")
T = t.TypeVar("T")
_T_co = t.TypeVar("_T_co", covariant=True, bound=t.Any)

rich_console = Console(theme=None)

__all__ = [
    "bentoml_cattr",
    "cached_property",
    "cached_contextmanager",
    "reserve_free_port",
    "LazyLoader",
    "validate_or_create_dir",
    "rich_console",
    "experimental",
    "compose",
]

_EXPERIMENTAL_APIS: set[str] = set()

logger = logging.getLogger(__name__)


def warn_experimental(api_name: str) -> None:
    """
    Warns the user that the given API is experimental.
    Make sure that if the API is not experimental anymore, this function call is removed.

    If 'api_name' requires string formatting, use %-formatting for optimization.

    Args:
        api_name: The name of the API that is experimental.
    """
    if api_name not in _EXPERIMENTAL_APIS:
        _EXPERIMENTAL_APIS.add(api_name)
        msg = "'%s' is an EXPERIMENTAL API and is currently not yet stable. Proceed with caution!"
        logger.warning(msg, api_name)


def experimental(
    f: t.Callable[P, t.Any] | None = None, *, api_name: str | None = None
) -> t.Callable[..., t.Any]:
    """
    Decorator to mark an API as experimental.

    If 'api_name' requires string formatting, use %-formatting for optimization.

    Args:
        api_name: The name of the API that is experimental.
    """
    if api_name is None:
        api_name = f.__name__ if inspect.isfunction(f) else repr(f)

    def decorator(func: t.Callable[..., t.Any]) -> t.Callable[P, t.Any]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> t.Any:
            warn_experimental(api_name)
            return func(*args, **kwargs)

        return wrapper

    if f is None:
        return decorator
    return decorator(f)


def add_experimental_docstring(f: t.Callable[P, t.Any]) -> t.Callable[P, t.Any]:
    """
    Decorator to add an experimental warning to the docstring of a function.
    """
    f.__doc__ = "[EXPERIMENTAL] " + (f.__doc__ if f.__doc__ is not None else "")
    return f


@overload
def first_not_none(*args: T | None, default: T) -> T:
    ...


@overload
def first_not_none(*args: T | None) -> T | None:
    ...


def first_not_none(*args: T | None, default: None | T = None) -> T | None:
    """
    Returns the first argument that is not None.
    """
    return next((arg for arg in args if arg is not None), default)


def normalize_labels_value(label: dict[str, t.Any] | None) -> dict[str, str] | None:
    if not label:
        return label
    if any(not isinstance(v, str) for v in label.values()):
        logger.warning(
            "'labels' should be a dict[str, str] and enforced by BentoML. Converting all values to string."
        )
    return {k: str(v) for k, v in label.items()}


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


def human_readable_size(size: t.Union[int, float], decimal_places: int = 2) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if size < 1024.0 or unit == "PiB":
            break
        size /= 1024.0
    else:
        raise ValueError("size is too large")
    return f"{size:.{decimal_places}f} {unit}"


def split_with_quotes(
    s: str,
    sep: str = ",",
    quote: str = '"',
    use_regex: bool = False,
) -> list[str]:
    """
    Split a string with quotes, e.g.:
    >>> split_with_quotes('a,b,"c,d",e')
    ['a', 'b', 'c,d', 'e']
    """
    if use_regex:
        assert (
            "(" not in sep and ")" not in sep
        ), "sep cannot contain '(' or ')' when using regex"
        reg = "({quote}[^{quote}]*{quote})|({sep})".format(
            quote=quote,
            sep=sep,
        )
    else:
        reg = "({quote}[^{quote}]*{quote})|({sep})".format(
            quote=re.escape(quote),
            sep=re.escape(sep),
        )
    raw_parts = re.split(reg, s)
    parts: list[str] = []
    part_begin = 0
    for i in range(0, len(raw_parts), 3):
        if i + 2 > len(raw_parts):
            parts.append("".join(filter(None, raw_parts[part_begin : i + 2])))
            continue
        if raw_parts[i + 2] is not None:
            parts.append("".join(filter(None, raw_parts[part_begin : i + 2])))
            part_begin = i + 3
            continue
    return parts


@contextlib.contextmanager
def reserve_free_port(
    host: str = "localhost",
    port: int | None = None,
    prefix: t.Optional[str] = None,
    max_retry: int = 50,
    enable_so_reuseport: bool = False,
) -> t.Iterator[int]:
    """
    detect free port and reserve until exit the context
    """
    import psutil

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if enable_so_reuseport:
        if psutil.WINDOWS:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        elif psutil.MACOS or psutil.FREEBSD:
            sock.setsockopt(socket.SOL_SOCKET, 0x10000, 1)  # SO_REUSEPORT_LB
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

            if sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 0:
                raise RuntimeError("Failed to set SO_REUSEPORT.") from None
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
                f"Cannot find free port with prefix {prefix} after {max_retry} retries."
            ) from None
    else:
        if port:
            sock.bind((host, port))
        else:
            sock.bind((host, 0))
    try:
        yield sock.getsockname()[1]
    finally:
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

    Example Usage: (To reuse the container based on the same image)

    .. code-block:: python

        @cached_contextmanager("{docker_image.id}")
        def start_docker_container_from_image(docker_image, timeout=60):
            container = ...
            yield container
            container.stop()
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


class compose:
    """
    Function composition: compose(f, g)(...) is equivalent to f(g(...)).
    Refer to https://github.com/mentalisttraceur/python-compose for original implementation.

    Args:
        *functions: Functions (or other callables) to compose.

    Raises:
        TypeError: If no arguments are given, or any argument is not callable.
    """

    def __init__(self: Self, *functions: F[t.Any]):
        if not functions:
            raise TypeError(f"{self!r} needs at least one argument.")
        _functions: list[F[t.Any]] = []
        for function in reversed(functions):
            if not callable(function):
                raise TypeError(f"{self!r} arguments must be callable.")
            if isinstance(function, compose):
                _functions.extend(function.functions)
            else:
                _functions.append(function)
        self.__wrapped__ = _functions[0]
        self._wrappers = tuple(_functions[1:])

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Call the composed function."""
        result = self.__wrapped__(*args, **kwargs)
        for function in self._wrappers:
            result = function(result)
        return result

    def __get__(self, obj: t.Any, typ_: type | None = None):
        """Get the composed function as a bound method."""
        wrapped = self.__wrapped__
        try:
            bind = type(wrapped).__get__
        except AttributeError:
            return self
        bound_wrapped = bind(wrapped, obj, typ_)
        if bound_wrapped is wrapped:
            return self
        bound_self = type(self)(bound_wrapped)
        bound_self._wrappers = self._wrappers
        return bound_self

    @_recursive_repr("<...>")
    def __repr__(self):
        return f"{self!r}({','.join(map(repr, reversed(self.functions)))})"

    @property
    def functions(self):
        """Read-only tuple of the composed callables, in order of execution."""
        return (self.__wrapped__,) + tuple(self._wrappers)


def is_async_callable(obj: t.Any) -> t.TypeGuard[t.Callable[..., t.Awaitable[t.Any]]]:
    # Borrowed from starlette._utils
    while isinstance(obj, functools.partial):
        obj = obj.func

    return asyncio.iscoroutinefunction(obj) or (
        callable(obj) and asyncio.iscoroutinefunction(obj.__call__)
    )
