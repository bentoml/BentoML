import uuid
import socket
import typing as t
import functools
import contextlib
from pathlib import Path

from ..types import PathType
from .lazy_loader import LazyLoader

C = t.TypeVar("C")
T = t.TypeVar("T")
_T_co = t.TypeVar("_T_co", covariant=True, bound=t.Any)


__all__ = [
    "cached_property",
    "cached_contextmanager",
    "reserve_free_port",
    "get_free_port",
    "catch_exceptions",
    "LazyLoader",
    "validate_or_create_dir",
]


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

    # TODO: use ParamSpec (3.10+): https://github.com/python/mypy/issues/8645
    def __call__(  # noqa: F811
        self, func: t.Callable[..., _T_co]
    ) -> t.Callable[..., t.Optional[_T_co]]:
        @functools.wraps(func)
        def _(*args: t.Any, **kwargs: t.Any) -> t.Optional[_T_co]:
            try:
                return func(*args, **kwargs)
            except self._catch_exc:
                if self._raises:
                    raise self._throw_exc(self._msg)
                return self._fallback

        return _


@contextlib.contextmanager
def reserve_free_port(host: str = "localhost") -> t.Iterator[int]:
    """
    detect free port and reserve until exit the context
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    port = sock.getsockname()[1]
    yield port
    sock.close()


def get_free_port(host: str = "localhost") -> int:
    """
    detect free port and reserve until exit the context
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    port: int = sock.getsockname()[1]
    sock.close()
    return port


class cached_property(t.Generic[C, T]):
    """A property that is only computed once per instance and then replaces
    itself with an ordinary attribute. Deleting the attribute resets the
    property.
    """

    def __init__(self, func: t.Callable[[C], T]):
        try:
            functools.update_wrapper(self, func)
        except AttributeError:
            pass
        self.func = func

    # pylint: disable=attribute-defined-outside-init
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj: C, cls: t.Type[C]) -> T:
        if obj is None:
            raise AttributeError(f"'{cls}' has no member '{self.name}'")
        value = obj.__dict__[self.name] = self.func(obj)
        return value


VT = t.TypeVar("VT")


class cached_contextmanager:
    """
    Just like contextlib.contextmanager, but will cache the yield value for the same
    arguments. When one instance of the contextmanager exits, the cache value will
    also be poped.
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
        self,
        func: t.Callable[..., t.Generator[VT, None, None]],
    ) -> t.Callable[..., t.ContextManager[VT]]:

        func_m = contextlib.contextmanager(func)

        @contextlib.contextmanager
        @functools.wraps(func)
        def _func(*args: t.Any, **kwargs: t.Any) -> t.Any:
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
