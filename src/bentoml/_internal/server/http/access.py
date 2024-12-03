from __future__ import annotations

import logging
from contextvars import ContextVar
from timeit import default_timer
from typing import TYPE_CHECKING
from typing import Sequence

if TYPE_CHECKING:
    from ... import external_typing as ext

REQ_CONTENT_LENGTH = "REQUEST_CONTENT_LENGTH"
REQ_CONTENT_TYPE = "REQUEST_CONTENT_TYPE"
RESP_CONTENT_LENGTH = "RESPONSE_CONTENT_LENGTH"
RESP_CONTENT_TYPE = "RESPONSE_CONTENT_TYPE"

CONTENT_LENGTH = b"content-length"
CONTENT_TYPE = b"content-type"

status: ContextVar[int] = ContextVar("ACCESS_LOG_STATUS_CODE")
request_content_length: ContextVar[bytes] = ContextVar(
    "ACCESS_LOG_REQ_CONTENT_LENGTH",
    default=b"",
)
request_content_type: ContextVar[bytes] = ContextVar(
    "ACCESS_LOG_REQ_CONTENT_TYPE",
    default=b"",
)
response_content_length: ContextVar[bytes] = ContextVar(
    "ACCESS_LOG_RESP_CONTENT_LENGTH",
    default=b"",
)
response_content_type: ContextVar[bytes] = ContextVar(
    "ACCESS_LOG_RESP_CONTENT_TYPE",
    default=b"",
)


class AccessLogMiddleware:
    """
    ASGI Middleware implementation that intercepts and decorates the send
    and receive callables to generate the BentoML access log.
    """

    def __init__(
        self,
        app: ext.ASGIApp,
        has_request_content_length: bool = False,
        has_request_content_type: bool = False,
        has_response_content_length: bool = False,
        has_response_content_type: bool = False,
        skip_paths: Sequence[str] = (),
    ) -> None:
        self.app = app
        self.has_request_content_length = has_request_content_length
        self.has_request_content_type = has_request_content_type
        self.has_response_content_length = has_response_content_length
        self.has_response_content_type = has_response_content_type
        self.logger = logging.getLogger("bentoml.access")
        self.skip_paths = skip_paths

    async def __call__(
        self,
        scope: ext.ASGIScope,
        receive: ext.ASGIReceive,
        send: ext.ASGISend,
    ) -> None:
        if not scope["type"].startswith(("http", "websocket")):
            await self.app(scope, receive, send)
            return

        start = default_timer()
        client = scope.get("client")
        scheme = scope["scheme"]
        method = scope.get("method", "")
        path = scope["path"]
        if path.startswith(tuple(self.skip_paths)):
            await self.app(scope, receive, send)
            return

        if client:
            address = f"{client[0]}:{client[1]}"
        else:
            address = "_"

        if self.has_request_content_length or self.has_request_content_type:
            for key, value in scope["headers"]:
                if key == CONTENT_LENGTH:
                    request_content_length.set(value)
                elif key == CONTENT_TYPE:
                    request_content_type.set(value)

        async def wrapped_receive() -> "ext.ASGIMessage":
            message = await receive()
            if message["type"] == "websocket.connect":
                self.logger.info(
                    "%s (scheme=%s,path=%s) - Client connected", address, scheme, path
                )
            elif message["type"] == "websocket.disconnect":
                self.logger.info(
                    "%s (scheme=%s,path=%s) - Client disconnected",
                    address,
                    scheme,
                    path,
                )
            return message

        async def wrapped_send(message: "ext.ASGIMessage") -> None:
            latency = max(default_timer() - start, 0) * 1000
            if message["type"] == "http.response.start":
                status.set(message["status"])
                if self.has_response_content_length or self.has_response_content_type:
                    for key, value in message["headers"]:
                        if key == CONTENT_LENGTH:
                            response_content_length.set(value)
                        elif key == CONTENT_TYPE:
                            response_content_type.set(value)
            elif message["type"] == "websocket.close":
                self.logger.info(
                    "%s (scheme=%s,path=%s) - Connection closed", address, scheme, path
                )
            elif message["type"] == "websocket.accept":
                self.logger.info(
                    "%s (scheme=%s,path=%s) - Connection established",
                    address,
                    scheme,
                    path,
                )

            elif message["type"] == "http.response.body":
                if "more_body" in message and message["more_body"]:
                    await send(message)
                    return

                request = [f"scheme={scheme}", f"method={method}", f"path={path}"]
                if self.has_request_content_type:
                    request.append(f"type={request_content_type.get().decode()}")
                if self.has_request_content_length:
                    request.append(f"length={request_content_length.get().decode()}")

                response = [f"status={status.get()}"]
                if self.has_response_content_type:
                    response.append(f"type={response_content_type.get().decode()}")
                if self.has_response_content_length:
                    response.append(f"length={response_content_length.get().decode()}")

                await send(message)
                self.logger.info(
                    "%s (%s) (%s) %.3fms",
                    address,
                    ",".join(request),
                    ",".join(response),
                    latency,
                )
                return

            await send(message)

        await self.app(scope, wrapped_receive, wrapped_send)
