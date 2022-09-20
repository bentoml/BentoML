from __future__ import annotations

import typing as t
import functools
import dataclasses
from typing import TYPE_CHECKING

from grpc import aio

if TYPE_CHECKING:
    from bentoml.grpc.types import Request
    from bentoml.grpc.types import Response
    from bentoml.grpc.types import RpcMethodHandler
    from bentoml.grpc.types import AsyncHandlerMethod
    from bentoml.grpc.types import HandlerCallDetails
    from bentoml.grpc.types import BentoServicerContext


@dataclasses.dataclass
class Context:
    usage: str
    accuracy_score: float


class AsyncContextInterceptor(aio.ServerInterceptor):
    def __init__(self, *, usage: str, accuracy_score: float) -> None:
        self.context = Context(usage=usage, accuracy_score=accuracy_score)
        self._record: set[str] = set()

    async def intercept_service(
        self,
        continuation: t.Callable[[HandlerCallDetails], t.Awaitable[RpcMethodHandler]],
        handler_call_details: HandlerCallDetails,
    ) -> RpcMethodHandler:
        from bentoml.grpc.utils import wrap_rpc_handler

        handler = await continuation(handler_call_details)

        if handler and (handler.response_streaming or handler.request_streaming):
            return handler

        def wrapper(behaviour: AsyncHandlerMethod[Response]):
            @functools.wraps(behaviour)
            async def new_behaviour(
                request: Request, context: BentoServicerContext
            ) -> Response | t.Awaitable[Response]:
                self._record.update(
                    {f"{self.context.usage}:{self.context.accuracy_score}"}
                )
                resp = await behaviour(request, context)
                context.set_trailing_metadata(
                    tuple(
                        [
                            (k, str(v).encode("utf-8"))
                            for k, v in dataclasses.asdict(self.context).items()
                        ]
                    )
                )
                return resp

            return new_behaviour

        return wrap_rpc_handler(wrapper, handler)
