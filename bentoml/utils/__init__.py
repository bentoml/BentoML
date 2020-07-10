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

import re
from io import StringIO
import socket
from contextlib import contextmanager
from functools import wraps
from urllib.parse import urlparse, uses_netloc, uses_params, uses_relative

_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard("")


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


def is_url(url):
    try:
        return urlparse(url).scheme in _VALID_URLS
    except Exception:  # pylint:disable=broad-except
        return False


def isidentifier(s):
    """
    Return true if string is in a valid python identifier format:

    https://docs.python.org/2/reference/lexical_analysis.html#identifiers
    """
    try:
        return s.isidentifier()
    except AttributeError:
        # str#isidentifier is only available in python 3
        return re.match(r"[A-Za-z_][A-Za-z_0-9]*\Z", s) is not None


def dump_to_yaml_str(yaml_dict):
    from ruamel.yaml import YAML

    yaml = YAML()
    string_io = StringIO()
    yaml.dump(yaml_dict, string_io)
    return string_io.getvalue()


def pb_to_yaml(message):
    from google.protobuf.json_format import MessageToDict

    message_dict = MessageToDict(message)
    return dump_to_yaml_str(message_dict)


def ProtoMessageToDict(protobuf_msg, **kwargs):
    from google.protobuf.json_format import MessageToDict

    if 'preserving_proto_field_name' not in kwargs:
        kwargs['preserving_proto_field_name'] = True

    return MessageToDict(protobuf_msg, **kwargs)


# This function assume the status is not status.OK
def status_pb_to_error_code_and_message(pb_status):
    from bentoml.yatai.proto import status_pb2

    assert pb_status.status_code != status_pb2.Status.OK
    error_code = status_pb2.Status.Code.Name(pb_status.status_code)
    error_message = pb_status.error_message
    return error_code, error_message


def cached_property(method):
    @property
    @wraps(method)
    def _m(self, *args, **kwargs):
        if not hasattr(self, '_cached_properties'):
            setattr(self, '_cached_properties', dict())
        if method.__name__ not in self._cached_properties:
            self._cached_properties[method.__name__] = method(self, *args, **kwargs)
        return self._cached_properties[method.__name__]

    return _m


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
