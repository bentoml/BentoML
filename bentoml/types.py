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

from dataclasses import dataclass, field
import io
import os
from typing import (
    Any,
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
import urllib
import uuid

from multidict import CIMultiDict
from werkzeug.formparser import parse_form_data
from werkzeug.http import parse_options_header

from bentoml.utils.dataclasses import json_serializer

BATCH_HEADER = "Bentoml-Is-Batch-Request"

# For non latin1 characters: https://tools.ietf.org/html/rfc8187
# Also https://github.com/benoitc/gunicorn/issues/1778
HEADER_CHARSET = 'latin1'

JSON_CHARSET = 'utf-8'


@json_serializer(fields=['uri', 'name'], compat=True)
@dataclass(frozen=False)
class FileLike:
    """
    An universal lazy-loading wrapper for file-like objects.
    It accepts URI, file path or bytes and provides interface like opened file object.

    Attributes
    ----------
    bytes : bytes, optional

    uri : str, optional
        The set of possible uris is:

        - ``file:///home/user/input.json``
        - ``http://site.com/input.csv`` (Not implemented)
        - ``https://site.com/input.csv`` (Not implemented)

    name : str, default None

    """

    bytes_: Optional[bytes] = None
    uri: Optional[str] = None
    name: Optional[str] = None

    _stream: Optional[BinaryIO] = None

    def __post_init__(self):
        if self.name is None:
            if self._stream is not None:
                self.name = getattr(self._stream, "name", None)
            elif self.uri is not None:
                p = urllib.parse.urlparse(self.uri)
                if p.scheme and p.scheme != "file":
                    raise NotImplementedError(
                        f"{self.__class__} now supports scheme 'file://' only"
                    )
                _, self.name = os.path.split(self.path)

    @property
    def path(self):
        r'''
        supports:

        /home/user/file
        C:\Python27\Scripts\pip.exe
        \\localhost\c$\WINDOWS\clock.avi
        \\networkstorage\homes\user

        https://stackoverflow.com/a/61922504/3089381
        '''
        parsed = urllib.parse.urlparse(self.uri)
        raw_path = urllib.request.url2pathname(urllib.parse.unquote(parsed.path))
        host = "{0}{0}{mnt}{0}".format(os.path.sep, mnt=parsed.netloc)
        path = os.path.abspath(os.path.join(host, raw_path))
        return path

    @property
    def stream(self):
        if self._stream is not None:
            pass
        elif self.bytes_ is not None:
            self._stream = io.BytesIO(self.bytes_)
        elif self.uri is not None:
            self._stream = open(self.path, "rb")
        else:
            return io.BytesIO()
        return self._stream

    def read(self, size=-1):
        # TODO: also write to log
        return self.stream.read(size)

    def seek(self, pos):
        return self.stream.seek(pos)

    def tell(self):
        return self.stream.tell()

    def close(self):
        if self._stream is not None:
            self._stream.close()

    def __del__(self):
        if getattr(self, "_stream", None) and not self._stream.closed:
            self._stream.close()


class HTTPHeaders(CIMultiDict):
    """
    A case insensitive mapping of HTTP headers' keys and values.
    It also parses several commonly used fields for easier access.

    Attributes
    ----------
    content_type : str
        The value of ``Content-Type``, for example:
        - ``application/json``
        - ``text/plain``
        - ``text/csv``

    charset : str
        The charset option of ``Content-Type``

    content_encoding : str
        The charset option of ``Content-Encoding``

    Methods
    -------
    from_dict : create a HTTPHeaders object from a dict

    from_sequence : create a HTTPHeaders object from a list/tuple

    """

    @property
    def content_type(self) -> str:
        return parse_options_header(self.get('content-type'))[0].lower()

    @property
    def charset(self) -> Optional[str]:
        _, options = parse_options_header(self.get('content-type'))
        charset = options.get('charset', None)
        assert charset is None or isinstance(charset, str)
        return charset

    @property
    def content_encoding(self) -> str:
        return parse_options_header(self.get('content-encoding'))[0].lower()

    @property
    def is_batch_input(self) -> Optional[bool]:
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


@dataclass
class HTTPRequest:
    """
    A common HTTP Request object.
    It also parses several commonly used fields for easier access.

    Attributes
    ----------
    headers : HTTPHeaders

    body : bytes

    """

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
            k: FileLike(_stream=f, name=f.filename) for k, f in files.items()
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
    body: Optional[bytes] = b""

    @classmethod
    def new(
        cls,
        status: int = 200,
        headers: Union[HTTPHeaders, dict, tuple, list] = None,
        body: bytes = b"",
    ):
        if headers is None:
            headers = HTTPHeaders()
        elif isinstance(headers, dict):
            headers = HTTPHeaders.from_dict(headers)
        elif isinstance(headers, (tuple, list)):
            headers = HTTPHeaders.from_sequence(headers)
        return cls(status, headers, body)

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
            status=self.status, headers=tuple(self.headers.items()), response=self.body
        )


# https://tools.ietf.org/html/rfc7159#section-3
JsonSerializable = Union[bool, None, Dict, List, int, float, str]

# https://docs.aws.amazon.com/lambda/latest/dg/python-handler.html
AwsLambdaEvent = Union[Dict, List, str, int, float, None]

Input = TypeVar("Input")
Output = TypeVar("Output")

ApiFuncArgs = TypeVar("ApiFuncArgs")
BatchApiFuncArgs = TypeVar("BatchApiFuncArgs")
ApiFuncReturnValue = TypeVar("ApiFuncReturnValue")
BatchApiFuncReturnValue = TypeVar("BatchApiFuncReturnValue")


@json_serializer(compat=True)
@dataclass
class InferenceResult(Generic[Output]):
    """
    The data structure that returned by BentoML API server.
    Contains result data and context like HTTP headers.
    """

    version: int = 0

    # payload
    data: Optional[Output] = None
    err_msg: str = ''

    # meta
    task_id: Optional[str] = None

    # context
    http_status: int = 501
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
        """
        Generate InferenceResults based on successful inference results and
        fallback results of discarded tasks.

        """
        iterable_results = iter(results)
        try:
            for task in tasks:
                if task.is_discarded:
                    assert task.error
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
    """
    The default InferenceResult when errors happened.
    """

    # context
    http_status: int = 500
    cli_status: int = 1


@json_serializer(compat=True)
@dataclass
class InferenceTask(Generic[Input]):
    """
    The data structure passed to the BentoML API server for inferring.
    Contains payload data and context like HTTP headers or CLI args.
    """

    version: int = 0

    # payload
    data: Optional[Input] = None
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
    inference_job_args: Optional[Mapping[str, Any]] = None

    def discard(self, err_msg="", **context):
        """
        Discard this task. All subsequent steps will be skipped.

        Parameters
        ----------
        err_msg: str
            The reason why this task got discarded. It would be the body of
            HTTP Response, a field in AWS lambda event or CLI stderr message.

        *other contexts
            Other contexts of the fallback ``InferenceResult``
        """
        self.is_discarded = True
        self.error = InferenceError(err_msg=err_msg, **context)
        return self
