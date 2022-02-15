import logging
from timeit import default_timer
from typing import List
from typing import TYPE_CHECKING
from contextvars import ContextVar

from starlette.middleware import Middleware

if TYPE_CHECKING:
    from .. import ext_typing as ext

REQ_CONTENT_LENGTH = "REQUEST_CONTENT_LENGTH"
REQ_CONTENT_TYPE = "REQUEST_CONTENT_TYPE"
RESP_CONTENT_LENGTH = "RESPONSE_CONTENT_LENGTH"
RESP_CONTENT_TYPE = "RESPONSE_CONTENT_TYPE"

CONTENT_LENGTH = b"content-length"
CONTENT_TYPE = b"content-type"

status: ContextVar[int] = ContextVar("ACCESS_LOG_STATUS_CODE")
request_content_length: ContextVar[bytes] = ContextVar(
    "ACCESS_LOG_REQ_CONTENT_LENGTH", default=b""
)
request_content_type: ContextVar[bytes] = ContextVar(
    "ACCESS_LOG_REQ_CONTENT_TYPE", default=b""
)
response_content_length: ContextVar[bytes] = ContextVar(
    "ACCESS_LOG_RESP_CONTENT_LENGTH", default=b""
)
response_content_type: ContextVar[bytes] = ContextVar(
    "ACCESS_LOG_RESP_CONTENT_TYPE", default=b""
)


class AccessLogMiddleware(Middleware):
    """
    ASGI Middleware implementation that intercepts and decorates the send
    and receive callables to generate the BentoML access log.
    """

    def __init__(self, app: "ext.ASGIApp", fields: List[str] = []) -> None:
        self.app = app
        self.fields = fields
        self.logger = logging.getLogger("bentoml.access")

    async def __call__(
        self,
        scope: "ext.ASGIScope",
        receive: "ext.ASGIReceive",
        send: "ext.ASGISend",
    ) -> None:
        if not scope["type"].startswith("http"):
            await self.app(scope, receive, send)
            return

        start = default_timer()
        client = scope["client"]
        scheme = scope["scheme"]
        method = scope["method"]
        path = scope["path"]

        if len(self.fields) > 0:
            for key, value in scope["headers"]:
                if key == CONTENT_LENGTH:
                    request_content_length.set(value)
                elif key == CONTENT_TYPE:
                    request_content_type.set(value)

        async def wrapped_send(message: "ext.ASGIMessage") -> None:
            if message["type"] == "http.response.start":
                status.set(message["status"])
                if len(self.fields) > 0:
                    for key, value in message["headers"]:
                        if key == CONTENT_LENGTH:
                            response_content_length.set(value)
                        elif key == CONTENT_TYPE:
                            response_content_type.set(value)

            elif message["type"] == "http.response.body":
                if client:
                    address = f"{client[0]}:{client[1]}"
                else:
                    address = "unknown_client"

                request = [f"scheme={scheme}", f"method={method}", f"path={path}"]
                if REQ_CONTENT_TYPE in self.fields:
                    request.append(f"type={request_content_type.get().decode()}")
                if REQ_CONTENT_LENGTH in self.fields:
                    request.append(f"length={request_content_length.get().decode()}")

                response = [f"status={status.get()}"]
                if RESP_CONTENT_TYPE in self.fields:
                    response.append(f"type={response_content_type.get().decode()}")
                if RESP_CONTENT_LENGTH in self.fields:
                    response.append(f"length={response_content_length.get().decode()}")

                latency = max(default_timer() - start, 0)

                self.logger.info(
                    "%s (%s) (%s) %.3fms",
                    address,
                    ",".join(request),
                    ",".join(response),
                    latency,
                )

            await send(message)

        await self.app(scope, receive, wrapped_send)
