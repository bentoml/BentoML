from __future__ import annotations

import io
import typing as t
import uuid
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from tempfile import SpooledTemporaryFile
from urllib.parse import unquote_plus

import multipart.multipart as multipart
from starlette.datastructures import FormData
from starlette.datastructures import Headers
from starlette.datastructures import MutableHeaders
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import Response

from ...exceptions import BentoMLException
from .http import set_cookies

if t.TYPE_CHECKING:
    from ..context import ServiceContext as Context

# Code below adapted from starlette's formparser. See the license: https://github.com/encode/starlette/blob/fc480890fe1f1e421746de303c6f8da1323e5626/LICENSE.md


class FormMessage(Enum):
    FIELD_START = 1
    FIELD_NAME = 2
    FIELD_DATA = 3
    FIELD_END = 4
    END = 5


@dataclass
class MultipartPart:
    content_disposition: bytes | None = None
    field_name: str = ""
    data: bytes = b""
    file: UploadFile | None = None
    item_headers: list[tuple[bytes, bytes]] = field(default_factory=list)


def _user_safe_decode(src: bytes, codec: str) -> str:
    try:
        return src.decode(codec)
    except (UnicodeDecodeError, LookupError):
        return src.decode("latin-1")


class MultiPartException(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


class FormParser:
    def __init__(self, headers: Headers, stream: t.AsyncGenerator[bytes, None]) -> None:
        assert (
            multipart is not None
        ), "The `python-multipart` library must be installed to use form parsing."
        self.headers = headers
        self.stream = stream
        self.messages: list[tuple[FormMessage, bytes]] = []

    def on_field_start(self) -> None:
        message = (FormMessage.FIELD_START, b"")
        self.messages.append(message)

    def on_field_name(self, data: bytes, start: int, end: int) -> None:
        message = (FormMessage.FIELD_NAME, data[start:end])
        self.messages.append(message)

    def on_field_data(self, data: bytes, start: int, end: int) -> None:
        message = (FormMessage.FIELD_DATA, data[start:end])
        self.messages.append(message)

    def on_field_end(self) -> None:
        message = (FormMessage.FIELD_END, b"")
        self.messages.append(message)

    def on_end(self) -> None:
        message = (FormMessage.END, b"")
        self.messages.append(message)

    async def parse(self) -> FormData:
        # Callbacks dictionary.
        callbacks = {
            "on_field_start": self.on_field_start,
            "on_field_name": self.on_field_name,
            "on_field_data": self.on_field_data,
            "on_field_end": self.on_field_end,
            "on_end": self.on_end,
        }

        # Create the parser.
        parser = multipart.QuerystringParser(callbacks)
        field_name = b""
        field_value = b""

        items: list[tuple[str, str | UploadFile]] = []

        # Feed the parser with data from the request.
        async for chunk in self.stream:
            if chunk:
                parser.write(chunk)
            else:
                parser.finalize()
            messages = list(self.messages)
            self.messages.clear()
            for message_type, message_bytes in messages:
                if message_type == FormMessage.FIELD_START:
                    field_name = b""
                    field_value = b""
                elif message_type == FormMessage.FIELD_NAME:
                    field_name += message_bytes
                elif message_type == FormMessage.FIELD_DATA:
                    field_value += message_bytes
                elif message_type == FormMessage.FIELD_END:
                    name = unquote_plus(field_name.decode("latin-1"))
                    value = unquote_plus(field_value.decode("latin-1"))
                    items.append((name, value))

        return FormData(items)


class MultiPartParser:
    max_file_size = 1024 * 1024

    def __init__(
        self,
        headers: Headers,
        stream: t.AsyncGenerator[bytes, None],
        *,
        max_files: int | float = 1000,
        max_fields: int | float = 1000,
    ) -> None:
        assert (
            multipart is not None
        ), "The `python-multipart` library must be installed to use form parsing."
        self.headers = headers
        self.stream = stream
        self.max_files = max_files
        self.max_fields = max_fields
        self.items: list[tuple[str, str | UploadFile]] = []
        self._current_files = 0
        self._current_fields = 0
        self._current_partial_header_name: bytes = b""
        self._current_partial_header_value: bytes = b""
        self._current_part = MultipartPart()
        self._charset: str = ""
        self._file_parts_to_write: list[tuple[MultipartPart, bytes]] = []
        self._file_parts_to_finish: list[MultipartPart] = []
        self._files_to_close_on_error: list[SpooledTemporaryFile[bytes]] = []

    def on_part_begin(self) -> None:
        self._current_part = MultipartPart()

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        message_bytes = data[start:end]
        if self._current_part.file is None:
            self._current_part.data += message_bytes
        else:
            self._file_parts_to_write.append((self._current_part, message_bytes))

    def on_part_end(self) -> None:
        if self._current_part.file is None:
            self.items.append(
                (
                    self._current_part.field_name,
                    _user_safe_decode(self._current_part.data, self._charset),
                )
            )
        else:
            self._file_parts_to_finish.append(self._current_part)
            # The file can be added to the items right now even though it's not
            # finished yet, because it will be finished in the `parse()` method, before
            # self.items is used in the return value.
            self.items.append((self._current_part.field_name, self._current_part.file))

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._current_partial_header_name += data[start:end]

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._current_partial_header_value += data[start:end]

    def on_header_end(self) -> None:
        field = self._current_partial_header_name.lower()
        if field == b"content-disposition":
            self._current_part.content_disposition = self._current_partial_header_value
        self._current_part.item_headers.append(
            (field, self._current_partial_header_value)
        )
        self._current_partial_header_name = b""
        self._current_partial_header_value = b""

    def on_headers_finished(self) -> None:
        assert self._current_part.content_disposition is not None
        _, options = multipart.parse_options_header(
            self._current_part.content_disposition
        )
        try:
            self._current_part.field_name = _user_safe_decode(
                bytes(options[b"name"]), self._charset
            )
        except KeyError:
            raise MultiPartException(
                'The Content-Disposition header field "name" must be ' "provided."
            )
        if b"filename" in options:
            self._current_files += 1
            if self._current_files > self.max_files:
                raise MultiPartException(
                    f"Too many files. Maximum number of files is {self.max_files}."
                )
            filename = _user_safe_decode(bytes(options[b"filename"]), self._charset)
            tempfile = SpooledTemporaryFile(max_size=self.max_file_size)
            self._files_to_close_on_error.append(tempfile)
            self._current_part.file = UploadFile(
                file=tempfile,
                filename=filename,
                headers=Headers(raw=self._current_part.item_headers),
            )
        else:
            self._current_fields += 1
            if self._current_fields > self.max_fields:
                raise MultiPartException(
                    f"Too many fields. Maximum number of fields is {self.max_fields}."
                )
            self._current_part.file = None

    def on_end(self) -> None:
        pass

    async def parse(self) -> FormData:
        # Parse the Content-Type header to get the multipart boundary.
        _, params = multipart.parse_options_header(self.headers["Content-Type"])
        charset = params.get(b"charset", "utf-8")
        if isinstance(charset, bytes):
            charset = charset.decode("latin-1")
        self._charset = str(charset)
        try:
            boundary = params[b"boundary"]
        except KeyError:
            raise MultiPartException("Missing boundary in multipart.")

        # Callbacks dictionary.
        callbacks = {
            "on_part_begin": self.on_part_begin,
            "on_part_data": self.on_part_data,
            "on_part_end": self.on_part_end,
            "on_header_field": self.on_header_field,
            "on_header_value": self.on_header_value,
            "on_header_end": self.on_header_end,
            "on_headers_finished": self.on_headers_finished,
            "on_end": self.on_end,
        }

        # Create the parser.
        parser = multipart.MultipartParser(boundary, callbacks)
        try:
            # Feed the parser with data from the request.
            async for chunk in self.stream:
                parser.write(chunk)
                # Write file data, it needs to use await with the UploadFile methods
                # that call the corresponding file methods *in a threadpool*,
                # otherwise, if they were called directly in the callback methods above
                # (regular, non-async functions), that would block the event loop in
                # the main thread.
                for part, data in self._file_parts_to_write:
                    assert part.file  # for type checkers
                    await part.file.write(data)
                for part in self._file_parts_to_finish:
                    assert part.file  # for type checkers
                    await part.file.seek(0)
                self._file_parts_to_write.clear()
                self._file_parts_to_finish.clear()
        except MultiPartException as exc:
            # Close all the files if there was an error.
            for file in self._files_to_close_on_error:
                file.close()
            raise exc

        parser.finalize()
        return FormData(self.items)


def file_body_to_message(f: UploadFile):
    async def res():
        return {
            "type": "http.request",
            "body": await f.read(),
        }

    return res


async def populate_multipart_requests(request: Request) -> t.Dict[str, Request]:
    content_type_header = request.headers.get("Content-Type")
    content_type, _ = multipart.parse_options_header(content_type_header)
    assert content_type in (b"multipart/form-data", b"multipart/mixed")
    stream = request.stream()
    multipart_parser = MultiPartParser(request.headers, stream)
    try:
        form = await multipart_parser.parse()
    except multipart.MultipartParseError:
        raise BentoMLException("Invalid multipart requests")

    reqs: dict[str, Request] = dict()
    for field_name, data in form.items():
        scope = dict(request.scope)
        if isinstance(data, UploadFile):
            ori_headers = dict(scope.get("headers", dict()))
            ori_headers = t.cast(t.Dict[bytes, bytes], ori_headers)
            ori_headers.update(dict(data.headers))
            scope["headers"] = list(ori_headers.items())
        if "headers" not in scope:
            scope["headers"] = []

        req = Request(scope)
        req._form = FormData([(field_name, data)])  # type: ignore  # using internal starlette APIs
        if isinstance(data, str):
            req._body = bytes(data, "utf-8")
        elif isinstance(data, UploadFile):
            req._receive = (  # type: ignore  # using internal starlette APIs
                file_body_to_message(data)
            )
        reqs[field_name] = req
    return reqs


def _get_disp_filename(headers: MutableHeaders) -> t.Optional[bytes]:
    if "content-disposition" in headers:
        _, options = multipart.parse_options_header(headers["content-disposition"])
        if b"filename" in options:
            return t.cast(bytes, options[b"filename"])
    return None


async def concat_to_multipart_response(
    responses: t.Mapping[str, Response], ctx: Context | None
) -> Response:
    boundary = uuid.uuid4().hex
    boundary_bytes = boundary.encode("latin1")

    writer = io.BytesIO()
    for field_name, resp in responses.items():
        writer.write(b"--%b\r\n" % boundary_bytes)

        # headers
        filename = _get_disp_filename(resp.headers)
        if filename:
            writer.write(
                b'Content-Disposition: form-data; name="%b"; filename="%b"\r\n'
                % (field_name.encode("latin1"), filename)
            )
        else:
            writer.write(
                b'Content-Disposition: form-data; name="%b"\r\n'
                % field_name.encode("latin1")
            )

        for header_key, header_value in resp.raw_headers:
            if header_key == b"content-disposition":
                continue
            writer.write(b"%b: %b\r\n" % (header_key, header_value))
        writer.write(b"\r\n")

        # body
        writer.write(resp.body)
        writer.write(b"\r\n")

    writer.write(b"--%b--\r\n" % boundary_bytes)

    if ctx is not None:
        res = Response(
            writer.getvalue(),
            headers=ctx.response.metadata,  # type: ignore (pyright thinks the type is dict[Unknown, Unknown])
            media_type=f"multipart/form-data; boundary={boundary}",
            status_code=ctx.response.status_code,
        )
        set_cookies(res, ctx.response.cookies)
        return res
    else:
        return Response(
            writer.getvalue(), media_type=f"multipart/form-data; boundary={boundary}"
        )
