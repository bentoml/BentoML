from __future__ import annotations

import abc
import asyncio
import base64
import io
import json
import typing as t

from starlette.requests import Request
from starlette.responses import Response

if t.TYPE_CHECKING:
    from starlette.types import Message


async def _consume_response(response: Response) -> bytes:
    # This will collect the response body into a buffer
    # and run until the background task is done
    buffer = io.BytesIO()
    scope: dict[str, t.Any] = {"type": "http"}

    def receive() -> t.Awaitable[Message]:
        # A fake receive that never returns
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        return fut

    async def send(message: Message) -> None:
        if message.get("type") != "http.response.body":
            return
        buffer.write(message.get("body", b""))

    await response(scope, receive, send)
    return buffer.getvalue()


class Serde(abc.ABC):
    @abc.abstractmethod
    async def serialize_request(self, request: Request) -> bytes:
        raise NotImplementedError

    @abc.abstractmethod
    async def deserialize_request(self, data: bytes) -> Request:
        raise NotImplementedError

    @abc.abstractmethod
    async def serialize_response(self, response: Response) -> bytes:
        raise NotImplementedError

    @abc.abstractmethod
    async def deserialize_response(self, data: bytes) -> Response:
        raise NotImplementedError


class JSONSerde(Serde):
    HEADERS_ENCODING = "iso-8859-1"

    def get_scope(self, scope: dict[str, t.Any]) -> dict[str, t.Any]:
        encodable_fields = [
            "type",
            "asgi",
            "http_version",
            "method",
            "scheme",
            "path",
            "root_path",
            "raw_path",
            "query_string",
            "headers",
        ]
        new_scope = {k: v for k, v in scope.items() if k in encodable_fields}
        return new_scope

    def json_encode(self, obj: t.Any) -> bytes:
        def default(obj: t.Any) -> t.Any:
            if isinstance(obj, bytes):
                return obj.decode(self.HEADERS_ENCODING)
            raise TypeError(
                f"Object of type {obj.__class__.__name__} is not JSON serializable"
            )

        return json.dumps(obj, separators=(",", ":"), default=default).encode()

    async def serialize_request(self, request: Request) -> bytes:
        request_dict = {
            "scope": self.get_scope(request.scope),
            "content": base64.b64encode(await request.body()).decode("ascii"),
        }
        return self.json_encode(request_dict)

    async def deserialize_request(self, data: bytes) -> Request:
        request_dict = json.loads(data)
        scope = request_dict["scope"]
        for field in ("raw_path", "query_string"):
            if field in scope:
                scope[field] = scope[field].encode(self.HEADERS_ENCODING)
        if "headers" in scope:
            scope["headers"] = [
                (k.encode(self.HEADERS_ENCODING), v.encode(self.HEADERS_ENCODING))
                for k, v in scope["headers"]
            ]
        request = Request(scope)
        request._body = base64.b64decode(request_dict["content"].encode("ascii"))
        return request

    async def serialize_response(self, response: Response) -> bytes:
        response_dict = {
            "status": response.status_code,
            "headers": list(response.headers.items()),
            "content": base64.b64encode(await _consume_response(response)).decode(
                "ascii"
            ),
        }
        return json.dumps(response_dict, separators=(",", ":")).encode()

    async def deserialize_response(self, data: bytes) -> Response:
        response_dict = json.loads(data)
        response = Response(
            content=base64.b64decode(response_dict["content"].encode("ascii")),
            status_code=response_dict["status"],
        )
        response.raw_headers = [
            tuple(map(str.encode, h)) for h in response_dict["headers"]
        ]
        return response
