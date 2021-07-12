# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import functools
import inspect
from io import StringIO
import os
import socket
import tarfile
from typing import Optional
from urllib.parse import urlparse, uses_netloc, uses_params, uses_relative

from google.protobuf.message import Message

from bentoml.utils.gcs import is_gcs_url
from bentoml.utils.lazy_loader import LazyLoader
from bentoml.utils.s3 import is_s3_url

_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard("")


__all__ = [
    "cached_property",
    "reserve_free_port",
    "is_url",
    "dump_to_yaml_str",
    "pb_to_yaml",
    "ProtoMessageToDict",
    "status_pb_to_error_code_and_message",
    "catch_exceptions",
    "cached_contextmanager",
    "resolve_bento_bundle_uri",
    "is_gcs_url",
    "is_s3_url",
    "archive_directory_to_tar",
    "resolve_bundle_path",
]

yatai_proto = LazyLoader("yatai_proto", globals(), "bentoml.yatai.proto")

DEFAULT_CHUNK_SIZE = 1024 * 8  # 8kb


class _Missing(object):
    def __repr__(self):
        return "no value"

    def __reduce__(self):
        return "_missing"


_missing = _Missing()


class cached_property(property):
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
    """

    # implementation detail: A subclass of python's builtin property
    # decorator, we override __get__ to check for a cached value. If one
    # chooses to invoke __get__ by hand the property will still work as
    # expected because the lookup logic is replicated in __get__ for
    # manual invocation.

    def __init__(
        self, func, name=None, doc=None
    ):  # pylint:disable=super-init-not-called
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __set__(self, obj, value):
        obj.__dict__[self.__name__] = value

    def __get__(self, obj, type=None):  # pylint:disable=redefined-builtin
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, _missing)
        if value is _missing:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value


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

    def __init__(self, cache_key_template=None):
        self._cache_key_template = cache_key_template
        self._cache = {}

    def __call__(self, func):
        func_m = contextlib.contextmanager(func)

        @contextlib.contextmanager
        @functools.wraps(func)
        def _func(*args, **kwargs):
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()
            if self._cache_key_template:
                cache_key = self._cache_key_template.format(**bound_args.arguments)
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
def reserve_free_port(host="localhost"):
    """
    detect free port and reserve until exit the context
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    port = sock.getsockname()[1]
    yield port
    sock.close()


def is_url(url: str) -> bool:
    try:
        return urlparse(url).scheme in _VALID_URLS
    except Exception:  # pylint:disable=broad-except
        return False


def dump_to_yaml_str(yaml_dict):
    from bentoml.utils.ruamel_yaml import YAML

    yaml = YAML()
    string_io = StringIO()
    yaml.dump(yaml_dict, string_io)
    return string_io.getvalue()


def pb_to_yaml(message: Message) -> str:
    from google.protobuf.json_format import MessageToDict

    message_dict = MessageToDict(message)
    return dump_to_yaml_str(message_dict)


def ProtoMessageToDict(protobuf_msg: Message, **kwargs) -> object:
    from google.protobuf.json_format import MessageToDict

    if "preserving_proto_field_name" not in kwargs:
        kwargs["preserving_proto_field_name"] = True

    return MessageToDict(protobuf_msg, **kwargs)


# This function assume the status is not status.OK
def status_pb_to_error_code_and_message(pb_status) -> (int, str):
    from bentoml.yatai.proto import status_pb2

    assert pb_status.status_code != status_pb2.Status.OK
    error_code = status_pb2.Status.Code.Name(pb_status.status_code)
    error_message = pb_status.error_message
    return error_code, error_message


class catch_exceptions(object):
    def __init__(self, exceptions, fallback=None):
        self.exceptions = exceptions
        self.fallback = fallback

    def __call__(self, func):
        @functools.wraps(func)
        def _(*args, **kwargs):
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
    from bentoml.yatai.client import get_yatai_client

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


def get_default_yatai_client():
    from bentoml.yatai.client import YataiClient

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
) -> (str, str):
    file_name = f'{tarfile_name}.tar'
    tarfile_path = os.path.join(tarfile_dir, file_name)
    with tarfile.open(tarfile_path, mode="w:gz") as tar:
        # Use arcname to prevent storing the full path to the bundle,
        # from source_dir/path/to/file to path/to/file
        tar.add(source_dir, arcname='')
    return tarfile_path, file_name
