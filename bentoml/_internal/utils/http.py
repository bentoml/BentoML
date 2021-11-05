import typing

import multipart
from multipart.multipart import parse_options_header
from starlette.formparsers import Headers, MultiPartMessage
from starlette.requests import Request


class MultiPartParser:
    """
    modified from https://github.com/encode/starlette/blob/6af5c515e0a896cbf3f86ee043b88f6c24200bcf/starlette/formparsers.py#L113
    """

    def __init__(
        self, headers: Headers, stream: typing.AsyncGenerator[bytes, None]
    ) -> None:
        assert (
            multipart is not None
        ), "The `python-multipart` library must be installed to use form parsing."
        self.headers = headers
        self.stream = stream
        self.messages: typing.List[typing.Tuple[MultiPartMessage, bytes]] = []

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

    async def parse(
        self,
    ) -> typing.List[typing.Tuple[typing.List[typing.Tuple[bytes, bytes]], bytes]]:
        # Parse the Content-Type header to get the multipart boundary.
        _, params = parse_options_header(self.headers["Content-Type"])
        charset = params.get(b"charset", "utf-8")
        if isinstance(charset, bytes):
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
        headers: typing.List[typing.Tuple[bytes, bytes]] = list()
        data = b""

        items = []

        # Feed the parser with data from the request.
        async for chunk in self.stream:
            parser.write(chunk)
            messages = list(self.messages)
            self.messages.clear()
            for message_type, message_bytes in messages:
                if message_type == MultiPartMessage.PART_BEGIN:
                    data = b""
                    headers = list()
                elif message_type == MultiPartMessage.HEADER_FIELD:
                    header_field += message_bytes
                elif message_type == MultiPartMessage.HEADER_VALUE:
                    header_value += message_bytes
                elif message_type == MultiPartMessage.HEADER_END:
                    headers.append((header_field.lower(), header_value))
                    header_field = b""
                    header_value = b""
                elif message_type == MultiPartMessage.HEADERS_FINISHED:
                    pass
                elif message_type == MultiPartMessage.PART_DATA:
                    data += message_bytes
                elif message_type == MultiPartMessage.PART_END:
                    items.append((headers, data))

        parser.finalize()
        return items


async def parse_multipart_to_multireq(request: Request) -> typing.List[Request]:
    content_type_header = request.headers.get("Content-Type")
    content_type, _ = parse_options_header(content_type_header)
    assert content_type == b"multipart/form-data"
    multipart_parser = MultiPartParser(request.headers, request.stream())
    part_infos = await multipart_parser.parse()

    reqs = []
    for headers, datas in part_infos:
        scope = dict(request.scope)
        ori_headers = dict(scope.get("headers", list()))
        ori_headers = typing.cast(typing.Dict[bytes, bytes], ori_headers)
        ori_headers.update(dict(headers))
        scope["headers"] = list(ori_headers.items())
        req = Request(scope)
        req._body = datas
        reqs.append(req)
    return reqs
