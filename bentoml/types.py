# Copyright 2020 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import io
import functools
from typing import NamedTuple, Union, Dict, List, Generic, TypeVar

from multidict import CIMultiDict


# For non latin1 characters: https://tools.ietf.org/html/rfc8187
# Also https://github.com/benoitc/gunicorn/issues/1778
HEADER_CHARSET = 'latin1'

JSON_CHARSET = 'utf-8'


class HTTPRequest(NamedTuple):
    '''
    headers: tuple of key value pairs in bytes
    data: str
    '''

    headers: tuple = tuple()
    body: bytes = b""

    @property
    @functools.lru_cache()
    def parsed_headers(self):
        return CIMultiDict(
            (k.decode(HEADER_CHARSET), v.decode(HEADER_CHARSET))
            for k, v in self.headers or tuple()
        )

    @classmethod
    def from_flask_request(cls, request):
        return cls(
            tuple(
                (k.encode(HEADER_CHARSET), v.encode(HEADER_CHARSET))
                for k, v in request.headers
            ),
            request.get_data(),
        )

    def to_flask_request(self):
        from werkzeug.wrappers import Request

        return Request.from_values(
            input_stream=io.BytesIO(self.body),
            content_length=len(self.body),
            headers=self.headers,
        )


class HTTPResponse(NamedTuple):
    status: int = 200
    headers: tuple = tuple()
    body: str = ""  # TODO(bojiang): bytes

    def to_flask_response(self):
        import flask

        return flask.Response(
            status=self.status, headers=self.headers, response=self.body
        )


# https://tools.ietf.org/html/rfc7159#section-3
JsonSerializable = Union[bool, None, Dict, List, int, float, str]

# https://docs.aws.amazon.com/lambda/latest/dg/python-handler.html
AwsLambdaEvent = Union[Dict, List, str, int, float, None]


Input = TypeVar("Input")
Output = TypeVar("Output")


class InferenceTask(Generic[Input]):
    def __init__(self, data: Input, context: dict = None):
        self.data = data
        self.context = context or {}
        self.is_discarded = False

    def discard(self, msg: str):
        self.is_discarded = True
        self.data = msg


class InferenceResult(Generic[Output]):
    is_error = False

    def __init__(self, data: Output, context: dict = None):
        self.data = data
        self.context = context or {}


class InferenceCollection(Generic[Input]):
    pass
