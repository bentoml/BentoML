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
    "generate_new_version_id",
    "catch_exceptions",
    "is_gcs_url",
    "is_s3_url",
    "LazyLoader",
]


def generate_new_version_id():
    """
    The default function for generating a new unique version string when saving a new
    bento or a new model
    """
    date_string = datetime.now().strftime("%Y%m%d")
    random_hash = uuid.uuid4().hex[:6].upper()

    # Example output: '20210910_D246ED'
    return f"{date_string}_{random_hash}"



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
