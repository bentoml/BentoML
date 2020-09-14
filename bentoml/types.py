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
import os
import urllib
import uuid
from dataclasses import dataclass, field
from typing import (
    BinaryIO,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from multidict import CIMultiDict
from werkzeug.formparser import parse_form_data
from werkzeug.http import parse_options_header

from bentoml import config
from bentoml.utils.dataclasses import json_serializer

BATCH_HEADER = config("apiserver").get("batch_request_header")

# For non latin1 characters: https://tools.ietf.org/html/rfc8187
# Also https://github.com/benoitc/gunicorn/issues/1778
HEADER_CHARSET = 'latin1'

JSON_CHARSET = 'utf-8'


@json_serializer(fields=['uri', 'name'], compat=True)
@dataclass(frozen=False)
class FileLike:
    stream: Optional[BinaryIO] = None
    bytes_: Optional[bytes] = None
    uri: Optional[str] = None
    name: Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            if self.stream is not None:
                self.name = getattr(self.stream, "name", None)
            elif self.uri is not None:
                p = urllib.parse.urlparse(self.uri)
                if p.scheme and p.scheme != "file":
                    raise NotImplementedError(
                        f"{self.__class__} now supports scheme 'file://' only"
                    )
                _, self.name = os.path.split(p.path)

    @property
    def path(self):
        return urllib.parse.urlparse(self.uri).path

    @property
    def _stream(self):
        if self.stream is not None:
            pass
        elif self.bytes_ is not None:
            self.stream = io.BytesIO(self.bytes_)
        elif self.uri is not None:
            self.stream = open(self.path, "rb")
        else:
            return io.BytesIO()
        return self.stream

    def read(self, size=-1):
        # TODO: also write to log
        return self._stream.read(size)

    def seek(self, pos):
        return self._stream.seek(pos)

    def tell(self):
        return self._stream.tell()

    def close(self):
        if self.stream is not None:
            self.stream.close()

    def __del__(self):
        if self.stream and not self.stream.closed:
            self.stream.close()


class HTTPHeaders(CIMultiDict):
    @property
    def content_type(self) -> str:
        return parse_options_header(self.get('content-type'))[0].lower()

    @property
    def content_encoding(self) -> str:
        return parse_options_header(self.get('content-encoding'))[0].lower()

    @property
    def is_batch_input(self) -> bool:
        hv = parse_options_header(self.get(BATCH_HEADER))[0].lower()
        return hv == "true" if hv else None

    @classmethod
    def from_dict(cls, d: Mapping[str, str]):
        return cls(d)

    @classmethod
    def from_sequence(cls, seq: Sequence[Tuple[str, str]]):
        return cls(seq)

    def to_json(self):
        return tuple(self.items())


@dataclass(frozen=False)
class HTTPRequest:
    '''
    headers: tuple of key value pairs in strs
    data: str
    '''

    headers: HTTPHeaders = HTTPHeaders()
    body: bytes = b""

    def __post_init__(self):
        if self.headers is None:
            self.headers = HTTPHeaders()
        elif isinstance(self.headers, dict):
            self.headers = HTTPHeaders.from_dict(self.headers)
        elif isinstance(self.headers, (tuple, list)):
            self.headers = HTTPHeaders.from_sequence(self.headers)

    @classmethod
    def parse_form_data(cls, self):
        if not self.body:
            return None, None, {}
        environ = {
            'wsgi.input': io.BytesIO(self.body),
            'CONTENT_LENGTH': len(self.body),
            'CONTENT_TYPE': self.headers.get('content-type', ''),
            'REQUEST_METHOD': 'POST',
        }
        stream, form, files = parse_form_data(environ, silent=False)
        wrapped_files = {
            k: FileLike(stream=f, name=f.filename) for k, f in files.items()
        }
        return stream, form, wrapped_files

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


@dataclass
class HTTPResponse:
    status: int = 200
    headers: HTTPHeaders = HTTPHeaders()
    body: bytes = b""

    def __post_init__(self):
        if self.headers is None:
            self.headers = HTTPHeaders()
        elif isinstance(self.headers, dict):
            self.headers = HTTPHeaders.from_dict(self.headers)
        elif isinstance(self.headers, (tuple, list)):
            self.headers = HTTPHeaders.from_sequence(self.headers)

    def to_flask_response(self):
        import flask

        return flask.Response(
            status=self.status, headers=self.headers.items(), response=self.body
        )


# https://tools.ietf.org/html/rfc7159#section-3
JsonSerializable = Union[bool, None, Dict, List, int, float, str]

# https://docs.aws.amazon.com/lambda/latest/dg/python-handler.html
AwsLambdaEvent = Union[Dict, List, str, int, float, None]

Input = TypeVar("Input")
Output = TypeVar("Output")

ApiFuncArgs = TypeVar("ApiFuncArgs")
ApiFuncReturnValue = TypeVar("ApiFuncReturnValue")


@json_serializer(compat=True)
@dataclass
class InferenceResult(Generic[Output]):
    version: int = 0

    # payload
    data: Output = None
    err_msg: str = ''

    # meta
    task_id: Optional[str] = None

    # context
    http_status: Optional[int] = None
    http_headers: HTTPHeaders = HTTPHeaders()
    aws_lambda_event: Optional[dict] = None
    cli_status: Optional[int] = 0

    def __post_init__(self):
        if self.http_headers is None:
            self.http_headers = HTTPHeaders()
        elif isinstance(self.http_headers, dict):
            self.http_headers = HTTPHeaders.from_dict(self.http_headers)
        elif isinstance(self.http_headers, (tuple, list)):
            self.http_headers = HTTPHeaders.from_sequence(self.http_headers)

    @classmethod
    def complete_discarded(
        cls, tasks: Iterable['InferenceTask'], results: Iterable['InferenceResult'],
    ) -> Iterator['InferenceResult']:
        iterable_results = iter(results)
        try:
            for task in tasks:
                if task.is_discarded:
                    yield task.error
                else:
                    yield next(iterable_results)
        except StopIteration:
            raise StopIteration(
                'The results does not match the number of tasks'
            ) from None


@json_serializer(compat=True)
@dataclass
class InferenceError(InferenceResult):
    # context
    http_status: int = 500
    cli_status: int = 1


@json_serializer(compat=True)
@dataclass
class InferenceTask(Generic[Input]):
    version: int = 0

    # payload
    data: Input = None
    error: Optional[InferenceResult] = None

    # meta
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_discarded: bool = False
    batch: Optional[int] = None

    # context
    http_method: Optional[str] = None
    http_headers: HTTPHeaders = HTTPHeaders()
    aws_lambda_event: Optional[dict] = None
    cli_args: Optional[Sequence[str]] = None

    def discard(self, err_msg="", **context):
        self.is_discarded = True
        self.error = InferenceError(err_msg=err_msg, **context)
        return self
