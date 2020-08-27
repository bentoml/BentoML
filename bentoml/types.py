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
    Tuple,
    Union,
    Dict,
    List,
    Sequence,
    Generic,
    TypeVar,
    Optional,
    Iterable,
    Iterator,
)
from dataclasses import dataclass

from multidict import CIMultiDict
from werkzeug.formparser import parse_form_data
from werkzeug.http import parse_options_header

# For non latin1 characters: https://tools.ietf.org/html/rfc8187
# Also https://github.com/benoitc/gunicorn/issues/1778
HEADER_CHARSET = 'latin1'

JSON_CHARSET = 'utf-8'


class ParsedHeaders(CIMultiDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.content_type = ""
        self.content_encoding = ""


@dataclass(frozen=True)
class HTTPRequest:
    '''
    headers: tuple of key value pairs in strs
    data: str
    '''

    headers: Sequence[Tuple[str, str]] = tuple()
    body: bytes = b""

    @property
    def parsed_headers(self) -> CIMultiDict:
        return self.parse_raw_headers(self.headers)

    @classmethod
    @functools.lru_cache()
    def parse_form_data(cls, self):
        if not self.body:
            return None, None, {}
        environ = {
            'wsgi.input': io.BytesIO(self.body),
            'CONTENT_LENGTH': len(self.body),
            'CONTENT_TYPE': self.parsed_headers.get('content-type', ''),
            'REQUEST_METHOD': 'POST',
        }
        stream, form, files = parse_form_data(environ, silent=False)
        for f in files.values():
            f.name = f.filename
        return stream, form, files

    @classmethod
    @functools.lru_cache()
    def parse_raw_headers(cls, raw_headers: Sequence[Tuple[str, str]]):
        headers_dict = ParsedHeaders(
            (k.lower(), v.lower()) for k, v in raw_headers or tuple()
        )
        headers_dict.content_type = parse_options_header(
            headers_dict.get('content-type')
        )[0]
        headers_dict.content_encoding = parse_options_header(
            headers_dict.get('content-encoding')
        )[0]
        return headers_dict

    @classmethod
    def from_flask_request(cls, request):
        return cls(
            tuple((k, v) for k, v in request.headers.items()), request.get_data(),
        )

    def to_flask_request(self):
        from werkzeug.wrappers import Request

        return Request.from_values(
            input_stream=io.BytesIO(self.body),
            content_length=len(self.body),
            headers=self.headers,
        )


@dataclass(frozen=True)
class HTTPResponse:
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

ApiFuncArgs = TypeVar("ApiFuncArgs")
ApiFuncReturnValue = TypeVar("ApiFuncReturnValue")


@dataclass(frozen=True)
class InferenceContext:
    task_id: Optional[str] = None

    # General
    err_msg: str = ''

    # HTTP
    http_method: Optional[str] = None
    http_status: Optional[int] = None
    http_headers: Optional[CIMultiDict] = None

    # AWS_LAMBDA
    aws_lambda_event: Optional[dict] = None

    # CLI
    cli_status: Optional[int] = 0
    cli_args: Optional[Sequence[str]] = None


@dataclass(frozen=True)
class DefaultErrorContext(InferenceContext):
    http_status: int = 500
    cli_status: int = 1


@dataclass(frozen=True)
class InferenceResult(Generic[Output]):
    data: Output = None
    context: InferenceContext = InferenceContext()

    @classmethod
    def complete_discarded(
        cls, tasks: Iterable['InferenceTask'], results: Iterable['InferenceResult'],
    ) -> Iterator['InferenceResult']:
        iterable_results = iter(results)
        try:
            for task in tasks:
                if task.is_discarded:
                    yield task.fallback_result
                else:
                    yield next(iterable_results)
        except StopIteration:
            raise StopIteration(
                'The results does not match the number of tasks'
            ) from None


@dataclass(frozen=False)
class InferenceTask(Generic[Input]):
    data: Input
    context: InferenceContext = InferenceContext()
    is_discarded: bool = False
    fallback_result: Optional[InferenceResult] = None

    def discard(self, err_msg="", **context):
        self.is_discarded = True
        self.fallback_result = InferenceResult(
            context=DefaultErrorContext(err_msg=err_msg, **context)
        )
