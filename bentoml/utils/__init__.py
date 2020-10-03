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

import socket
from contextlib import contextmanager
from functools import wraps
from io import StringIO
from urllib.parse import urlparse, uses_netloc, uses_params, uses_relative

from google.protobuf.message import Message
from werkzeug.utils import cached_property

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
]


@contextmanager
def reserve_free_port(host='localhost'):
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

    if 'preserving_proto_field_name' not in kwargs:
        kwargs['preserving_proto_field_name'] = True

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
        @wraps(func)
        def _(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except self.exceptions:
                return self.fallback

        return _
