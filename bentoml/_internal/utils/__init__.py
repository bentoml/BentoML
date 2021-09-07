import contextlib
import functools
import inspect
import os
import socket
import tarfile
import uuid
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from urllib.parse import urlparse, uses_netloc, uses_params, uses_relative

if TYPE_CHECKING:
    from mypy.typeshed.stdlib.contextlib import _GeneratorContextManager

from .gcs import is_gcs_url
from .lazy_loader import LazyLoader
from .s3 import is_s3_url

_T = TypeVar("_T")
_V = TypeVar("_V")

_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard("")

DEFAULT_CHUNK_SIZE = 1024 * 8  # 8kb


__all__ = [
    "reserve_free_port",
    "get_free_port",
    "is_url",
    "generate_new_version_id",
    "catch_exceptions",
    "cached_contextmanager",
    "cached_property",
    "resolve_bento_bundle_uri",
    "is_gcs_url",
    "is_s3_url",
    "archive_directory_to_tar",
    "resolve_bundle_path",
    "LazyLoader",
]


def _yield_first_val(iterable):
    if isinstance(iterable, tuple):
        yield iterable[0]
    elif isinstance(iterable, str):
        yield iterable
    else:
        yield from iterable


def flatten_list(lst) -> List[str]:
    if not isinstance(lst, list):
        raise AttributeError
    return [k for i in lst for k in _yield_first_val(i)]


def generate_new_version_id():
    return f'{datetime.now().strftime("%Y%m%d")}_{uuid.uuid4().hex[:6].upper()}'


class _Missing(object):
    def __repr__(self) -> str:
        return "no value"

    def __reduce__(self) -> str:
        return "_missing"


_missing = _Missing()


class cached_property(Generic[_T, _V], property):
    """A decorator that converts a function into a lazy property. The
    function wrapped is called the first time to retrieve the result
    and then that calculated result is used the next time you access
    the value:

    .. code-block:: python

        class Foo(object):

            @cached_property
            def foo(self):
                # calculate something important here
                return 42

    The class has to have a `__dict__` in order for this property to
    work.

    Implementation detail: A subclass of python's builtin property
    decorator, we override __get__ to check for a cached value. If one
    chooses to invoke __get__ by hand the property will still work as
    expected because the lookup logic is replicated in __get__ for
    manual invocation.
    """

    def __init__(
        self,
        func: Callable[[_T], _V],
        name: Optional[str] = None,
        doc: Optional[str] = None,
    ):  # pylint:disable=super-init-not-called
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __set__(self, obj: _T, value: _V) -> None:
        obj.__dict__[self.__name__] = value

    @overload
    def __get__(  # pylint: disable=redefined-builtin
        self, obj: None, type: Optional[Type[_T]] = None
    ) -> "cached_property":
        ...

    @overload
    def __get__(  # pylint: disable=redefined-builtin
        self, obj: _T, type: Optional[Type[_T]] = None
    ) -> _V:
        ...

    def __get__(  # pylint:disable=redefined-builtin
        self, obj: Optional[_T], type: Optional[Type[_T]] = None
    ) -> Union["cached_property", _V]:
        if obj is None:
            return self
        value: _V = obj.__dict__.get(self.__name__, _missing)
        if value is _missing:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value


class cached_contextmanager(Generic[_T]):
    """
    Just like contextlib.contextmanager, but will cache the yield value for the same
    arguments. When one instance of the contextmanager exits, the cache value will
    also be popped.

    (To reuse the container based on the same image)
    Example Usage::

        @cached_contextmanager("{docker_image.id}")
        def start_docker_container_from_image(docker_image, timeout=60):
            container = ...
            yield container
            container.stop()
    """

    def __init__(self, cache_key_template: Optional[str] = None) -> None:
        self._cache_key_template = cache_key_template
        self._cache: Dict[Union[str, Tuple], _T] = {}

    # TODO: use ParamSpec 3.10: https://github.com/python/mypy/issues/8645.
    #   One possible solution is to use typing_extensions >=3.8
    def __call__(
        self, func: Callable[..., Iterator[_T]]
    ) -> Callable[..., "_GeneratorContextManager[_T]"]:
        func_m = contextlib.contextmanager(func)

        @contextlib.contextmanager
        @functools.wraps(func)
        def _func(*args: Any, **kwargs: Any) -> Iterator[_T]:
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()
            if self._cache_key_template:
                cache_key: Union[str, Tuple] = self._cache_key_template.format(
                    **bound_args.arguments
                )
            else:
                cache_key = tuple(bound_args.arguments.values())
            if cache_key in self._cache:
                yield self._cache[cache_key]
            else:
                with func_m(*args, **kwargs) as value:
                    self._cache[cache_key] = value
                    yield value
                    self._cache.pop(cache_key)

        return _func


class catch_exceptions(Generic[_T], object):
    def __init__(
        self,
        catch_exc: Union[Type[BaseException], Tuple[Type[BaseException], ...]],
        throw_exc: Union[Type[BaseException], Tuple[Type[BaseException], ...]],
        msg: Optional[str] = "",
        fallback: Optional[_T] = None,
        raises: Optional[bool] = True,
    ) -> None:
        self._catch_exc = catch_exc
        self._throw_exc = throw_exc
        self._msg = msg
        self._fallback = fallback
        self._raises = raises

    # TODO: use ParamSpec (3.10+): https://github.com/python/mypy/issues/8645
    def __call__(self, func: Callable[..., _T]) -> Callable[..., Optional[_T]]:
        @functools.wraps(func)
        def _(*args: Any, **kwargs: Any) -> Optional[_T]:
            try:
                return func(*args, **kwargs)
            except self._catch_exc:
                if self._raises:
                    raise self._throw_exc(self._msg)
                return self._fallback

        return _


@contextlib.contextmanager
def reserve_free_port(host: str = "localhost") -> Iterator[int]:
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


def is_url(url: str) -> bool:
    try:
        return urlparse(url).scheme in _VALID_URLS
    except Exception:  # pylint:disable=broad-except
        return False


def resolve_bundle_path(
    bento: str,
    pip_installed_bundle_path: Optional[str] = None,
    yatai_url: Optional[str] = None,
) -> str:
    from bentoml.exceptions import BentoMLException

    if pip_installed_bundle_path:
        assert (
            bento is None
        ), "pip installed BentoService commands should not have Bento argument"
        return pip_installed_bundle_path

    if os.path.isdir(bento) or is_s3_url(bento) or is_gcs_url(bento):
        # saved_bundle already support loading local, s3 path and gcs path
        return bento
    else:
        raise BentoMLException(
            f'BentoService "{bento}" not found - either specify the file path of '
            f"the BentoService saved bundle, or the BentoService id in the form of "
            f'"name:version"'
        )


def resolve_bento_bundle_uri(bento_pb):
    if bento_pb.uri.s3_presigned_url:
        # Use s3 presigned URL for downloading the repository if it is presented
        return bento_pb.uri.s3_presigned_url
    if bento_pb.uri.gcs_presigned_url:
        return bento_pb.uri.gcs_presigned_url
    else:
        return bento_pb.uri.uri


def archive_directory_to_tar(
    source_dir: str, tarfile_dir: str, tarfile_name: str
) -> Tuple[str, str]:
    file_name = f"{tarfile_name}.tar"
    tarfile_path = os.path.join(tarfile_dir, file_name)
    with tarfile.open(tarfile_path, mode="w:gz") as tar:
        # Use arcname to prevent storing the full path to the bundle,
        # from source_dir/path/to/file to path/to/file
        tar.add(source_dir, arcname="")
    return tarfile_path, file_name
