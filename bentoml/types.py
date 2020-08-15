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
from typing import (
    NamedTuple,
    Tuple,
    Union,
    Dict,
    List,
    Generic,
    TypeVar,
    Optional,
    Iterable,
    Iterator,
)

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


UserArgs = TypeVar("UserArgs")
UserReturnValue = TypeVar("UserReturnValue")


# class UserArgs(OrderedDict):
# '''
# >>> print(*InputPair(x=[1, 2, 3], y=[4, 5, 6]))
# [1, 2, 3] [4, 5, 6]
# '''

# def __iter__(self):
# return iter(self.values())


class InferenceContext(NamedTuple):
    task_id: Optional[str] = None
    err_msg: str = ''

    http_method: Optional[str] = None
    http_status: Optional[int] = None
    http_headers: Optional[CIMultiDict] = None

    aws_lambda_event: Optional[dict] = None

    cli_status: Optional[int] = 0
    cli_args: Optional[Tuple[str]] = None


class DefaultErrorContext(InferenceContext):
    http_status: int = 500
    cli_status: int = 1


class InferenceTask(Generic[Input]):
    def __init__(self, data: Input, context: InferenceContext = None):
        self.data = data
        self.context = context or InferenceContext()
        self.is_discarded = False

    def discard(self, err_msg="", **context):
        self.is_discarded = True
        self.context = DefaultErrorContext(err_msg=err_msg, **context)


class InferenceResult(Generic[Output]):
    def __init__(self, data: Output = None, context: InferenceContext = None):
        self.data = data
        self.context = context or InferenceContext()

    @classmethod
    def complete_discarded(
        cls, tasks: Iterable[InferenceTask], results: Iterable['InferenceResult'],
    ) -> Iterator['InferenceResult']:
        iresults = iter(results)
        try:
            for task in tasks:
                if task.is_discarded:
                    yield cls(None, context=DefaultErrorContext(*task.context))
                else:
                    yield next(iresults)
        except StopIteration:
            raise StopIteration(
                'The results does not match the number of tasks'
            ) from None
