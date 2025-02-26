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
import typing as t
import warnings
from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from reprlib import recursive_repr as _recursive_repr
from typing import TYPE_CHECKING
from typing import overload

from ..types import LazyType

if TYPE_CHECKING:
    import attr
    from starlette.applications import Starlette
    from typing_extensions import Self

    from ..types import MetadataDict
    from ..types import MetadataType

    P = t.ParamSpec("P")
    F = t.Callable[P, t.Any]


C = t.TypeVar("C")
T = t.TypeVar("T")


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
def first_not_none(*args: T | None, default: T) -> T: ...


@overload
def first_not_none(*args: T | None) -> T | None: ...


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
        assert "(" not in sep and ")" not in sep, (
            "sep cannot contain '(' or ')' when using regex"
        )
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


def get_original_func(obj: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
    """
    Get the original function from a decorated function
    """
    if getattr(obj, "__is_bentoml_api_func__", False):
        obj = obj.func  # type: ignore
    while isinstance(obj, functools.partial):
        obj = obj.func
    return obj


def is_async_callable(obj: t.Any) -> t.TypeGuard[t.Callable[..., t.Awaitable[t.Any]]]:
    obj = get_original_func(obj)

    return asyncio.iscoroutinefunction(obj) or (
        callable(obj) and asyncio.iscoroutinefunction(obj.__call__)
    )


def dict_filter_none(d: dict[str, t.Any]) -> dict[str, t.Any]:
    return {k: v for k, v in d.items() if v is not None}


CONTROL_CODES = re.compile(
    r"""
    \x07|                  # BELL
    \r|                    # CARRIAGE_RETURN
    \x1b\[H|               # HOME
    \x1b\[2J|              # CLEAR
    \x1b\[?1049h|          # ENABLE_ALT_SCREEN
    \x1b\[?1049l|          # DISABLE_ALT_SCREEN
    \x1b\[?25h|            # SHOW_CURSOR
    \x1b\[?25l|            # HIDE_CURSOR
    \x1b\[\d+A|            # CURSOR_UP
    \x1b\[\d+B|            # CURSOR_DOWN
    \x1b\[\d+C|            # CURSOR_FORWARD
    \x1b\[\d+D|            # CURSOR_BACKWARD
    \x1b\[\d+G|            # CURSOR_MOVE_TO_COLUMN
    \x1b\[\d+K|            # ERASE_IN_LINE
    \x1b\[\d+;\d+H|        # CURSOR_MOVE_TO
    \x1b\]0;.+?\x07        # SET_WINDOW_TITLE
""",
    flags=re.VERBOSE,
)


def filter_control_codes(line: str) -> str:
    return CONTROL_CODES.sub("", line)


def with_app_arg(func: t.Callable[[], T]) -> t.Callable[[Starlette], T]:
    @functools.wraps(func)
    def wrapper(app: Starlette) -> T:
        return func()

    return wrapper


class BentoMLDeprecationWarning(DeprecationWarning):
    pass


warnings.simplefilter("default", BentoMLDeprecationWarning)


def warn_deprecated(message: str, stacklevel: int = 2) -> None:
    warnings.warn(
        message, category=BentoMLDeprecationWarning, stacklevel=stacklevel + 1
    )


def deprecated(
    name: str = "", deprecated_since: str = "1.4", suggestion: str = ""
) -> t.Callable[[C], C]:
    def decorator(func: t.Callable[P, T]) -> t.Callable[P, T]:
        obj_name = name or func.__name__

        class _DeprecatedMixin:
            def __init__(self, *args: P.args, **kwargs: P.kwargs) -> None:
                warn_deprecated(
                    f"`{obj_name}` is deprecated since BentoML v{deprecated_since} and will be removed in a future version."
                    + (f" {suggestion}" if suggestion else "")
                )
                super().__init__(*args, **kwargs)

        if inspect.isclass(func):
            return type(func.__name__, (_DeprecatedMixin, func), {})

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            warn_deprecated(
                f"`{obj_name}` is deprecated since BentoML v{deprecated_since} and will be removed in a future version."
                + (f" {suggestion}" if suggestion else "")
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def is_jupyter() -> bool:  # pragma: no cover
    """Check if we're running in a Jupyter notebook."""
    try:
        get_ipython  # type: ignore[name-defined]
    except NameError:
        return False
    ipython = get_ipython()  # noqa: F821 # type: ignore[name-defined]
    shell = ipython.__class__.__name__
    if (
        "google.colab" in str(ipython.__class__)
        or os.getenv("DATABRICKS_RUNTIME_VERSION")
        or shell == "ZMQInteractiveShell"
    ):
        return True  # Jupyter notebook or qtconsole
    elif shell == "TerminalInteractiveShell":
        return False  # Terminal running IPython
    else:
        return False  # Other type (?)


def deep_merge(base: t.Any, update: t.Any) -> dict[t.Any, t.Any]:
    """Merge two dictionaries recursively in place. The base dict will be updated."""
    if not isinstance(base, dict) or not isinstance(update, dict):
        return update
    for k, v in update.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = deep_merge(base[k], v)
        else:
            base[k] = v
    return base
