import io
import uuid
import typing as t

import multipart.multipart as multipart
from starlette.requests import Request
from starlette.responses import Response
from starlette.formparsers import MultiPartMessage
from starlette.datastructures import Headers
from starlette.datastructures import MutableHeaders

from ...exceptions import BentoMLException

_ItemsBody = t.List[t.Tuple[str, t.List[t.Tuple[bytes, bytes]], bytes]]


def user_safe_decode(src: bytes, codec: str) -> str:
    try:
        return src.decode(codec)
    except (UnicodeDecodeError, LookupError):
        return src.decode("latin-1")


class MultiPartParser:
    """
    An modified version of starlette MultiPartParser.
    """

    def __init__(self, headers: Headers, stream: t.AsyncGenerator[bytes, None]) -> None:
        assert (
            multipart is not None
        ), "The `python-multipart` library must be installed to use form parsing."
        self.headers: Headers = headers
        self.stream = stream
        self.messages: t.List[t.Tuple[MultiPartMessage, bytes]] = list()

    def on_part_begin(self) -> None:
        message = (MultiPartMessage.PART_BEGIN, b"")
        self.messages.append(message)

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        message = (MultiPartMessage.PART_DATA, data[start:end])
        self.messages.append(message)

    def on_part_end(self) -> None:
        message = (MultiPartMessage.PART_END, b"")
        self.messages.append(message)

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        message = (MultiPartMessage.HEADER_FIELD, data[start:end])
        self.messages.append(message)

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        message = (MultiPartMessage.HEADER_VALUE, data[start:end])
        self.messages.append(message)

    def on_header_end(self) -> None:
        message = (MultiPartMessage.HEADER_END, b"")
        self.messages.append(message)

    def on_headers_finished(self) -> None:
        message = (MultiPartMessage.HEADERS_FINISHED, b"")
        self.messages.append(message)

    def on_end(self) -> None:
        message = (MultiPartMessage.END, b"")
        self.messages.append(message)

    async def parse(self) -> _ItemsBody:
        # Parse the Content-Type header to get the multipart boundary.
        _, params = multipart.parse_options_header(self.headers["Content-Type"])
        params = t.cast(t.Dict[bytes, bytes], params)
        charset = params.get(b"charset", b"utf-8")
        charset = charset.decode("latin-1")
        boundary = params.get(b"boundary")

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
        header_field = b""
        header_value = b""
        field_name = ""

        data = b""

        items: _ItemsBody = []
        headers: t.List[t.Tuple[bytes, bytes]] = []

        # Feed the parser with data from the request.
        async for chunk in self.stream:
            parser.write(chunk)
            messages = list(self.messages)
            self.messages.clear()
            for message_type, message_bytes in messages:
                if message_type == MultiPartMessage.PART_BEGIN:
                    field_name = ""
                    data = b""
                    headers = list()
                elif message_type == MultiPartMessage.HEADER_FIELD:  # type: ignore
                    header_field += message_bytes
                elif message_type == MultiPartMessage.HEADER_VALUE:  # type: ignore
                    header_value += message_bytes
                elif message_type == MultiPartMessage.HEADER_END:  # type: ignore
                    field = header_field.lower()
                    if field == b"content-disposition":
                        _, options = multipart.parse_options_header(header_value)
                        options = t.cast(t.Dict[bytes, bytes], options)
                        field_name = user_safe_decode(options[b"name"], charset)
                    elif field == b"bentoml-payload-field":
                        field_name = user_safe_decode(header_value, charset)
                    else:
                        headers.append((field, header_value))
                    header_field = b""
                    header_value = b""
                elif message_type == MultiPartMessage.HEADERS_FINISHED:  # type: ignore
                    assert (
                        field_name
                    ), "`Content-Disposition` is not available in headers"
                elif message_type == MultiPartMessage.PART_DATA:  # type: ignore
                    data += message_bytes
                elif message_type == MultiPartMessage.PART_END:  # type: ignore
                    items.append((field_name, headers, data))

        parser.finalize()
        return items


async def populate_multipart_requests(request: Request) -> t.Dict[str, Request]:
    content_type_header = request.headers.get("Content-Type")
    content_type, _ = multipart.parse_options_header(content_type_header)
    assert content_type in (b"multipart/form-data", b"multipart/mixed")
    stream = t.cast(t.AsyncGenerator[bytes, None], request.stream())
    multipart_parser = MultiPartParser(request.headers, stream)
    try:
        form = await multipart_parser.parse()
    except multipart.MultipartParseError:
        raise BentoMLException("Invalid multipart requests")

    reqs = dict()  # type: t.Dict[str, Request]
    for field_name, headers, data in form:
        scope = dict(request.scope)
        ori_headers = dict(scope.get("headers", list()))
        ori_headers = t.cast(t.Dict[bytes, bytes], ori_headers)
        ori_headers.update(dict(headers))
        scope["headers"] = list(ori_headers.items())
        req = Request(scope)
        req._body = data
        reqs[field_name] = req
    return reqs


def _get_disp_filename(headers: MutableHeaders) -> t.Optional[bytes]:
    if "content-disposition" in headers:
        _, options = multipart.parse_options_header(headers["content-disposition"])
        if b"filename" in options:
            return t.cast(bytes, options[b"filename"])
    return None


async def concat_to_multipart_responses(
    responses: t.Mapping[str, Response]
) -> Response:
    boundary = uuid.uuid4().hex
    headers = {"content-type": f"multipart/form-data; boundary={boundary}"}

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

    return Response(writer.getvalue(), headers=headers)
