import binascii
import os
import typing as t
from collections import defaultdict

import multipart.multipart as multipart  # noqa
from starlette.datastructures import MutableHeaders
from starlette.formparsers import _user_safe_decode  # noqa
from starlette.formparsers import Headers, MultiPartMessage
from starlette.formparsers import MultiPartParser as _StarletteMultiPartParser
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from ...exceptions import BentoMLException

_ItemsBody = t.TypeVar(
    "_ItemsBody",
    bound=t.List[t.Tuple[str, t.List[t.Tuple[bytes, bytes]], bytes]],
)


class MultiPartParser(_StarletteMultiPartParser):
    """
    An modified version of starlette MultiPartParser.
    """

    headers: Headers
    stream: t.AsyncGenerator[bytes, None]

    def on_end(self) -> None:
        message = (MultiPartMessage.END, b"")
        self.messages.append(message)

    async def parse(self) -> _ItemsBody:
        # Parse the Content-Type header to get the multipart boundary.
        content_type, params = multipart.parse_options_header(
            self.headers["Content-Type"]
        )
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
        content_disposition = None
        field_name = ""

        # content_type = b""
        # file: t.Optional[UploadFile] = None
        data = b""

        items = t.cast(_ItemsBody, list())
        headers: t.List[t.Tuple[bytes, bytes]] = list()

        # Feed the parser with data from the request.
        async for chunk in self.stream:
            parser.write(chunk)
            messages = list(self.messages)
            self.messages.clear()
            for message_type, message_bytes in messages:
                if message_type == MultiPartMessage.PART_BEGIN:
                    content_disposition = None
                    # content_type = b''
                    data = b""
                    headers = list()
                elif message_type == MultiPartMessage.HEADER_FIELD:
                    header_field += message_bytes
                elif message_type == MultiPartMessage.HEADER_VALUE:
                    header_value += message_bytes
                elif message_type == MultiPartMessage.HEADER_END:
                    field = header_field.lower()
                    if field == b"content-disposition":
                        content_disposition = header_value
                    # elif field == b"content-type":
                    #     content_type = header_value
                    else:
                        headers.append((field, header_value))
                    header_field = b""
                    header_value = b""
                elif message_type == MultiPartMessage.HEADERS_FINISHED:
                    _, options = multipart.parse_options_header(content_disposition)
                    options = t.cast(t.Dict[bytes, bytes], options)
                    field_name = _user_safe_decode(options[b"name"], charset)
                    # if b"filename" in options:
                    #     filename = _user_safe_decode(options[b'filename'], charset)
                    #     file = UploadFile(
                    #         filename=filename,
                    #         content_type = content_type.decode('latin-1')
                    #     )
                    # else:
                    #     file = None
                elif message_type == MultiPartMessage.PART_DATA:
                    data += message_bytes
                    # if not file:
                    #     data += message_bytes
                    # else:
                    #     await file.write(message_bytes)
                elif message_type == MultiPartMessage.PART_END:
                    items.append((field_name, headers, data))
                    # if not file:
                    #     items.append((field_name, headers, data))
                    # else:
                    #     await file.seek(0)
                    #     items.append((field_name, file, headers))

        parser.finalize()
        return items


async def populate_multipart_requests(request: Request) -> t.Dict[str, Request]:
    content_type_header = request.headers.get("Content-Type")
    content_type, _ = multipart.parse_options_header(content_type_header)
    assert content_type == b"multipart/form-data"
    multipart_parser = MultiPartParser(request.headers, request.stream())
    try:
        form = await multipart_parser.parse()
    except multipart.MultipartParseError:
        raise BentoMLException("Invalid multipart requests")

    # NOTE: This is the equivalent of form = await request.form()
    request._form = form  # noqa

    reqs = dict()
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


async def concat_multipart_responses(
    responses: t.List[t.Tuple[str, t.Union[Response, StreamingResponse]]]
) -> Response:
    boundary = binascii.hexlify(os.urandom(16)).decode("ascii")

    content_type = "multipart/form-data; boundary=%s" % boundary
    content = b""
    header = list()
    for field_name, resp in responses:
        if isinstance(resp, StreamingResponse):
            async for chunk in resp.body_iterator:
                if not isinstance(chunk, bytes):
                    chunk = chunk.encode(resp.charset)
                content += chunk
            print(content)
        else:
            content += resp.body
        header.append(resp.headers)

    # body = (
    #     "".join("--%s\r\n"
    #             "Content-Disposition: form-data; name=\"%s\"\r\n"
    #             "\r\n"
    #             "%s\r\n" % (boundary, field, value)
    #             for field, value in fields.items()) +
    #     "--%s--\r\n" % boundary
    # )
    print(header)
    print(content)
    return Response(content=content, headers=header, media_type="multipart/form-data")
