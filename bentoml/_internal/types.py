import io
import os
import urllib
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, Mapping, Optional, Sequence, Tuple, Union

import attr
from multidict import CIMultiDict
from werkzeug.formparser import parse_form_data
from werkzeug.http import parse_options_header

from bentoml.exceptions import BentoMLException

from .utils.dataclasses import json_serializer

BATCH_HEADER = "Bentoml-Is-Batch-Request"

# For non latin1 characters: https://tools.ietf.org/html/rfc8187
# Also https://github.com/benoitc/gunicorn/issues/1778
HEADER_CHARSET = "latin1"

JSON_CHARSET = "utf-8"

PathType = Union[str, os.PathLike]

GenericDictType = Dict[str, Any]  # TODO:


@attr.s
class BentoTag:
    name = attr.ib(type=str)
    version = attr.ib(type=str)

    def __str__(self):
        return f"{self.name}:{self.version}"

    @classmethod
    def from_str(cls, tag_str: str) -> "BentoTag":
        try:
            name, version = tag_str.split(":")
            if not version:
                # in case users mistakenly define "bento:"
                raise BentoMLException(
                    f"{tag_str} contains leading ':'. Maybe you "
                    f"meant to use `{tag_str}:latest`?"
                )
            return cls(name, version)
        except ValueError:
            raise BentoMLException(f"Invalid {cls.__name__} {tag_str}")


@json_serializer(fields=["uri", "name"], compat=True)
@dataclass(frozen=False)
class FileLike:
    """
    An universal lazy-loading wrapper for file-like objects.
    It accepts URI, file path or bytes and provides interface like opened file object.

    Class attributes:

    - bytes (`bytes`, `optional`):
    - uri (`str`, `optional`):
        The set of possible uris is:

        - :code:`file:///home/user/input.json`
        - :code:`http://site.com/input.csv` (Not implemented)
        - :code:`https://site.com/input.csv` (Not implemented)

    - name (`str`, `optional`, default to :obj:`None`)

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
        r"""
        supports:

            /home/user/file
            C:\Python27\Scripts\pip.exe
            \\localhost\c$\WINDOWS\clock.avi
            \\networkstorage\homes\user

        .. note::
            https://stackoverflow.com/a/61922504/3089381
        """
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

    Class attributes:

    - content_type (`str`):
        The value of ``Content-Type``, for example:

        - :code:`application/json`
        - :code:`text/plain`
        - :code:`text/csv`

    - charset (`str`):
        The charset option of ``Content-Type``

    - content_encoding (`str`):
        The charset option of ``Content-Encoding``

    Class contains the following method:

    - from_dict : create a HTTPHeaders object from a dict

    - from_sequence : create a HTTPHeaders object from a list/tuple
    """

    @property
    def content_type(self) -> str:
        return parse_options_header(self.get("content-type"))[0].lower()

    @property
    def charset(self) -> Optional[str]:
        _, options = parse_options_header(self.get("content-type"))
        charset = options.get("charset", None)
        assert charset is None or isinstance(charset, str)
        return charset

    @property
    def content_encoding(self) -> str:
        return parse_options_header(self.get("content-encoding"))[0].lower()

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

    Class attributes:

    - headers (`HTTPHeaders`)

     - body (`bytes`)
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
            "wsgi.input": io.BytesIO(self.body),
            "CONTENT_LENGTH": len(self.body),
            "CONTENT_TYPE": self.headers.get("content-type", ""),
            "REQUEST_METHOD": "POST",
        }
        stream, form, files = parse_form_data(environ, silent=False)
        wrapped_files = {
            k: FileLike(_stream=f, name=f.filename) for k, f in files.items()
        }
        return stream, form, wrapped_files

    @classmethod
    def from_flask_request(cls, request):
        return cls(
            tuple((k, v) for k, v in request.headers.items()),
            request.get_data(),
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
