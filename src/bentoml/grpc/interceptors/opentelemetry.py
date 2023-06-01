from __future__ import annotations

import typing as t
import logging
import functools
from typing import TYPE_CHECKING
from contextlib import asynccontextmanager

from simple_di import inject
from simple_di import Provide
from opentelemetry import trace
from opentelemetry.context import attach
from opentelemetry.context import detach
from opentelemetry.propagate import extract
from opentelemetry.trace.status import Status
from opentelemetry.trace.status import StatusCode
from opentelemetry.semconv.trace import SpanAttributes

from bentoml.grpc.utils import import_grpc
from bentoml.grpc.utils import wrap_rpc_handler
from bentoml.grpc.utils import GRPC_CONTENT_TYPE
from bentoml.grpc.utils import parse_method_name
from bentoml._internal.utils.pkg import get_pkg_version
from bentoml._internal.configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    import grpc
    from grpc import aio
    from grpc.aio._typing import MetadataKey
    from grpc.aio._typing import MetadataType
    from grpc.aio._typing import MetadataValue
    from opentelemetry.trace import Span
    from opentelemetry.sdk.trace import TracerProvider

    from bentoml.grpc.types import Request
    from bentoml.grpc.types import Response
    from bentoml.grpc.types import RpcMethodHandler
    from bentoml.grpc.types import AsyncHandlerMethod
    from bentoml.grpc.types import HandlerCallDetails
    from bentoml.grpc.types import BentoServicerContext
else:
    grpc, aio = import_grpc()

logger = logging.getLogger(__name__)


class _OpenTelemetryServicerContext(aio.ServicerContext["Request", "Response"]):
    def __init__(self, servicer_context: BentoServicerContext, active_span: Span):
        self._servicer_context = servicer_context
        self._active_span = active_span
        self._code = grpc.StatusCode.OK
        self._details = ""
        super().__init__()

    def __getattr__(self, attr: str) -> t.Any:
        return getattr(self._servicer_context, attr)

    async def read(self) -> Request:
        return await self._servicer_context.read()

    async def write(self, message: Response) -> None:
        return await self._servicer_context.write(message)

    def trailing_metadata(self) -> aio.Metadata:
        return self._servicer_context.trailing_metadata()  # type: ignore (unfinished type)

    def auth_context(self) -> t.Mapping[str, t.Iterable[bytes]]:
        return self._servicer_context.auth_context()

    def peer_identity_key(self) -> str | None:
        return self._servicer_context.peer_identity_key()

    def peer_identities(self) -> t.Iterable[bytes] | None:
        return self._servicer_context.peer_identities()

    def peer(self) -> str:
        return self._servicer_context.peer()

    def disable_next_message_compression(self) -> None:
        self._servicer_context.disable_next_message_compression()

    def set_compression(self, compression: grpc.Compression) -> None:
        return self._servicer_context.set_compression(compression)

    def invocation_metadata(self) -> aio.Metadata | None:
        return self._servicer_context.invocation_metadata()

    def set_trailing_metadata(self, trailing_metadata: MetadataType) -> None:
        self._servicer_context.set_trailing_metadata(trailing_metadata)

    async def send_initial_metadata(self, initial_metadata: MetadataType) -> None:
        return await self._servicer_context.send_initial_metadata(initial_metadata)

    async def abort(
        self,
        code: grpc.StatusCode,
        details: str = "",
        trailing_metadata: MetadataType = tuple(),
    ) -> None:
        self._code = code
        self._details = details
        self._active_span.set_attribute(
            SpanAttributes.RPC_GRPC_STATUS_CODE, code.value[0]
        )
        self._active_span.set_status(
            Status(status_code=StatusCode.ERROR, description=f"{code}:{details}")
        )
        return await self._servicer_context.abort(
            code, details=details, trailing_metadata=trailing_metadata
        )

    def set_code(self, code: grpc.StatusCode) -> None:
        self._code = code
        details = self._details or code.value[1]
        self._active_span.set_attribute(
            SpanAttributes.RPC_GRPC_STATUS_CODE, code.value[0]
        )
        if code != grpc.StatusCode.OK:
            self._active_span.set_status(
                Status(status_code=StatusCode.ERROR, description=f"{code}:{details}")
            )
        return self._servicer_context.set_code(code)

    def code(self) -> grpc.StatusCode:
        return self._code

    def set_details(self, details: str) -> None:
        self._details = details
        if self._code != grpc.StatusCode.OK:
            self._active_span.set_status(
                Status(
                    status_code=StatusCode.ERROR, description=f"{self._code}:{details}"
                )
            )
        return self._servicer_context.set_details(details)

    def details(self) -> str:
        return self._details


# Since opentelemetry doesn't provide an async implementation for the server interceptor,
# we will need to create an async implementation ourselves.
# By doing this we will have more control over how to handle span and context propagation.
#
# Until there is a solution upstream, this implementation is sufficient for our needs.
class AsyncOpenTelemetryServerInterceptor(aio.ServerInterceptor):
    @inject
    def __init__(
        self,
        *,
        tracer_provider: TracerProvider = Provide[BentoMLContainer.tracer_provider],
        schema_url: str | None = None,
    ):
        self._tracer = tracer_provider.get_tracer(
            "opentelemetry.instrumentation.grpc",
            get_pkg_version("opentelemetry-instrumentation-grpc"),
            schema_url=schema_url,
        )

    @asynccontextmanager
    async def set_remote_context(
        self, servicer_context: BentoServicerContext
    ) -> t.AsyncGenerator[None, None]:
        metadata = servicer_context.invocation_metadata()
        if metadata:
            md: dict[MetadataKey, MetadataValue] = {m.key: m.value for m in metadata}
            ctx = extract(md)
            token = attach(ctx)
            try:
                yield
            finally:
                detach(token)
        else:
            yield

    def start_span(
        self,
        method_name: str,
        context: BentoServicerContext,
        set_status_on_exception: bool = False,
    ) -> t.ContextManager[Span]:
        attributes: dict[str, str | bytes] = {
            SpanAttributes.RPC_SYSTEM: "grpc",
            SpanAttributes.RPC_GRPC_STATUS_CODE: grpc.StatusCode.OK.value[0],
        }

        # method_name shouldn't be none, otherwise
        # it will never reach this point.
        method_rpc, _ = parse_method_name(method_name)
        attributes.update(
            {
                SpanAttributes.RPC_METHOD: method_rpc.method,
                SpanAttributes.RPC_SERVICE: method_rpc.fully_qualified_service,
            }
        )

        # add some attributes from the metadata
        metadata = context.invocation_metadata()
        if metadata:
            dct: dict[str, str | bytes] = dict(metadata)
            if "user-agent" in dct:
                attributes["rpc.user_agent"] = dct["user-agent"]

        # get trailing metadata
        trailing_metadata: MetadataType | None = context.trailing_metadata()
        if trailing_metadata:
            trailing = dict(trailing_metadata)
            attributes["rpc.content_type"] = trailing.get(
                "content-type", GRPC_CONTENT_TYPE
            )

        # Split up the peer to keep with how other telemetry sources
        # do it. This looks like:
        # * ipv6:[::1]:57284
        # * ipv4:127.0.0.1:57284
        # * ipv4:10.2.1.1:57284,127.0.0.1:57284
        #
        # the process ip and port would be [::1] 57284
        try:
            ipv4_addr = context.peer().split(",")[0]
            ip, port = ipv4_addr.split(":", 1)[1].rsplit(":", 1)
            attributes.update(
                {
                    SpanAttributes.NET_PEER_IP: ip,
                    SpanAttributes.NET_PEER_PORT: port,
                }
            )
            # other telemetry sources add this, so we will too
            if ip in ("[::1]", "127.0.0.1"):
                attributes[SpanAttributes.NET_PEER_NAME] = "localhost"
        except IndexError:
            logger.warning("Failed to parse peer address '%s'", context.peer())

        return self._tracer.start_as_current_span(
            name=method_name,
            kind=trace.SpanKind.SERVER,
            attributes=attributes,
            set_status_on_exception=set_status_on_exception,
        )

    async def intercept_service(
        self,
        continuation: t.Callable[[HandlerCallDetails], t.Awaitable[RpcMethodHandler]],
        handler_call_details: HandlerCallDetails,
    ) -> RpcMethodHandler:
        handler = await continuation(handler_call_details)
        method_name = handler_call_details.method

        # Currently not support streaming RPCs.
        if handler and (handler.response_streaming or handler.request_streaming):
            return handler

        def wrapper(behaviour: AsyncHandlerMethod[Response]):
            @functools.wraps(behaviour)
            async def new_behaviour(
                request: Request, context: BentoServicerContext
            ) -> Response | t.Awaitable[Response]:

                async with self.set_remote_context(context):
                    with self.start_span(method_name, context) as span:
                        # wrap context
                        wrapped_context = _OpenTelemetryServicerContext(context, span)

                        # And now we run the actual RPC.
                        try:
                            return await behaviour(request, wrapped_context)
                        except Exception as e:
                            # We are interested in uncaught exception, otherwise
                            # it will be handled by gRPC.
                            if type(e) != Exception:
                                span.record_exception(e)
                            raise e

            return new_behaviour

        return wrap_rpc_handler(wrapper, handler)
