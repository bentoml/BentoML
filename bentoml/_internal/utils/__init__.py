import contextlib
import functools
import inspect
import os
import socket
import tarfile
from io import StringIO
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

from google.protobuf.message import Message

if TYPE_CHECKING:
    from mypy.typeshed.stdlib.contextlib import _GeneratorContextManager
    from bentoml._internal.yatai_client import YataiClient

from .gcs import is_gcs_url
from .lazy_loader import LazyLoader
from .s3 import is_s3_url

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_F = TypeVar("_F", bound=Callable[..., Any])

T = TypeVar("T")
V = TypeVar("V")


def _yield_first_val(iterable):
    if isinstance(iterable, tuple):
        yield iterable[0]
    elif isinstance(iterable, str):
        yield iterable
    else:
        for i in iterable:
            yield i


def _flatten_list(lst) -> List[str]:
    if not isinstance(lst, list):
        raise AttributeError
    return [k for i in lst for k in _yield_first_val(i)]


_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard("")

__all__ = [
    "reserve_free_port",
    "get_free_port",
    "is_url",
    "dump_to_yaml_str",
    "pb_to_yaml",
    "ProtoMessageToDict",
    "status_pb_to_error_code_and_message",
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

yatai_proto = LazyLoader("yatai_proto", globals(), "bentoml.yatai.proto")

DEFAULT_CHUNK_SIZE = 1024 * 8  # 8kb


class _Missing(object):
    def __repr__(self) -> str:
        return "no value"

    def __reduce__(self) -> str:
        return "_missing"


_missing = _Missing()


class cached_property(Generic[T, V], property):
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
        func: Callable[[T], V],
        name: Optional[str] = None,
        doc: Optional[str] = None,
    ):  # pylint:disable=super-init-not-called
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __set__(self, obj: T, value: V) -> None:
        obj.__dict__[self.__name__] = value

    @overload
    def __get__(  # pylint: disable=redefined-builtin
        self, obj: None, type: Optional[Type[T]] = None
    ) -> "cached_property":
        ...

    @overload
    def __get__(  # pylint: disable=redefined-builtin
        self, obj: T, type: Optional[Type[T]] = None
    ) -> V:
        ...

    def __get__(  # pylint:disable=redefined-builtin
        self, obj: Optional[T], type: Optional[Type[T]] = None
    ) -> Union["cached_property", V]:
        if obj is None:
            return self
        value: V = obj.__dict__.get(self.__name__, _missing)
        if value is _missing:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value


class cached_contextmanager(Generic[T]):
    """
    Just like contextlib.contextmanager, but will cache the yield value for the same
    arguments. When one instance of the contextmanager exits, the cache value will
    also be poped.

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
        self._cache: Dict[Union[str, Tuple], T] = {}

    # TODO: use ParamSpec 3.10: https://github.com/python/mypy/issues/8645
    def __call__(
        self, func: Callable[..., Iterator[T]]
    ) -> Callable[..., "_GeneratorContextManager[T]"]:
        func_m = contextlib.contextmanager(func)

        @contextlib.contextmanager
        @functools.wraps(func)
        def _func(*args: Any, **kwargs: Any) -> Iterator[T]:
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


def dump_to_yaml_str(yaml_dict: Dict) -> str:
    from ..utils.ruamel_yaml import YAML

    yaml = YAML()
    string_io = StringIO()
    yaml.dump(yaml_dict, string_io)
    return string_io.getvalue()


def pb_to_yaml(message: Message) -> str:
    from google.protobuf.json_format import MessageToDict

    message_dict = MessageToDict(message)
    return dump_to_yaml_str(message_dict)


def ProtoMessageToDict(protobuf_msg: Message, **kwargs: Any) -> object:
    from google.protobuf.json_format import MessageToDict

    if "preserving_proto_field_name" not in kwargs:
        kwargs["preserving_proto_field_name"] = True

    return MessageToDict(protobuf_msg, **kwargs)


# This function assume the status is not status.OK
def status_pb_to_error_code_and_message(pb_status) -> Tuple[int, str]:
    from ..yatai_client.proto import status_pb2

    assert pb_status.status_code != status_pb2.Status.OK
    error_code = status_pb2.Status.Code.Name(pb_status.status_code)
    error_message = pb_status.error_message
    return error_code, error_message


class catch_exceptions(Generic[T], object):
    def __init__(
        self,
        exceptions: Union[Type[BaseException], Tuple[Type[BaseException], ...]],
        fallback: Optional[T] = None,
    ) -> None:
        self.exceptions = exceptions
        self.fallback = fallback

    # TODO: use ParamSpec (3.10+): https://github.com/python/mypy/issues/8645
    def __call__(self, func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @functools.wraps(func)
        def _(*args: Any, **kwargs: Any) -> Optional[T]:
            try:
                return func(*args, **kwargs)
            except self.exceptions:
                return self.fallback

        return _


def resolve_bundle_path(
    bento: str,
    pip_installed_bundle_path: Optional[str] = None,
    yatai_url: Optional[str] = None,
) -> str:
    from bentoml.exceptions import BentoMLException

    from ..yatai_client import get_yatai_client

    if pip_installed_bundle_path:
        assert (
            bento is None
        ), "pip installed BentoService commands should not have Bento argument"
        return pip_installed_bundle_path

    if os.path.isdir(bento) or is_s3_url(bento) or is_gcs_url(bento):
        # saved_bundle already support loading local, s3 path and gcs path
        return bento

    elif ":" in bento:
        # assuming passing in BentoService in the form of Name:Version tag
        yatai_client = get_yatai_client(yatai_url)
        bento_pb = yatai_client.repository.get(bento)
        return resolve_bento_bundle_uri(bento_pb)
    else:
        raise BentoMLException(
            f'BentoService "{bento}" not found - either specify the file path of '
            f"the BentoService saved bundle, or the BentoService id in the form of "
            f'"name:version"'
        )


def get_default_yatai_client() -> "YataiClient":
    from bentoml._internal.yatai_client import YataiClient

    return YataiClient()


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
