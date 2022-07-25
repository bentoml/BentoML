from __future__ import annotations

import sys
import typing as t
import asyncio
import logging
from abc import ABCMeta
from abc import abstractmethod
from typing import TYPE_CHECKING

from grpc import aio

from bentoml.exceptions import BentoMLException

from ....utils.grpc import grpc_status_code
from ....utils.grpc import get_factory_and_method

if TYPE_CHECKING:
    from ..types import RequestType
    from ..types import ResponseType
    from ..types import RpcMethodHandler
    from ..types import HandlerCallDetails
    from ..types import BentoServicerContext

logger = logging.getLogger(__name__)

# verbatim sources with modification from https://github.com/d5h-foss/grpc-interceptor
# added better typing that fits with BentoService signatures.
class AsyncServerInterceptor(aio.ServerInterceptor, metaclass=ABCMeta):
    """
    Base class for BentoService server-side interceptors.

    To implement, subclass this one and override ``intercept`` method.
    """

    @abstractmethod
    async def intercept(
        self,
        method: t.Callable[
            [RequestType, BentoServicerContext], t.Awaitable[ResponseType]
        ],
        request: RequestType,
        context: BentoServicerContext,
        method_name: str,
    ) -> t.Any:
        response_or_iterator = method(request, context)
        if hasattr(response_or_iterator, "__aiter__"):
            return response_or_iterator
        else:
            return await response_or_iterator

    # Implementation of grpc.ServerInterceptor, do not override.
    async def intercept_service(
        self,
        continuation: t.Callable[[HandlerCallDetails], t.Awaitable[RpcMethodHandler]],
        handler_call_details: HandlerCallDetails,
    ) -> RpcMethodHandler:
        """Implementation of grpc.aio.ServerInterceptor.
        This is not part of the grpc_interceptor.AsyncServerInterceptor API, but must
        have a public name. Do not override it, unless you know what you're doing.
        """
        next_handler = await continuation(handler_call_details)
        handler_factory, next_handler_method = get_factory_and_method(next_handler)

        if next_handler.response_streaming:

            async def invoke_intercept_method(  # type: ignore (function redefinition)
                request: RequestType, context: BentoServicerContext
            ) -> t.AsyncGenerator[ResponseType, None]:
                method_name = handler_call_details.method
                coroutine_or_asyncgen = self.intercept(
                    next_handler_method, request, context, method_name
                )

                # Async server streaming handlers return async_generator, because they
                # use the async def + yield syntax. However, this is NOT a coroutine
                # and hence is not awaitable. This can be a problem if the interceptor
                # ignores the individual streaming response items and simply returns the
                # result of method(request, context). In that case the interceptor IS a
                # coroutine, and hence should be awaited. In both cases, we need
                # something we can iterate over so that THIS function is an
                # async_generator like the actual RPC method.
                if asyncio.iscoroutine(coroutine_or_asyncgen):
                    asyncgen_or_none = await coroutine_or_asyncgen
                    # If a handler is using the read/write API, it will return None.
                    if not asyncgen_or_none:
                        return
                    asyncgen = asyncgen_or_none
                else:
                    asyncgen = coroutine_or_asyncgen

                async for r in asyncgen:
                    yield r

        else:

            async def invoke_intercept_method(
                request: RequestType, context: BentoServicerContext
            ) -> t.Awaitable[ResponseType]:
                method_name = handler_call_details.method
                return await self.intercept(
                    next_handler_method,
                    request,
                    context,
                    method_name,
                )

        return handler_factory(
            invoke_intercept_method,
            request_deserializer=next_handler.request_deserializer,
            response_serializer=next_handler.response_serializer,
        )


class ExceptionHandlerInterceptor(AsyncServerInterceptor):
    """An async interceptor that handles exceptions raised via BentoService."""

    async def handle_exception(
        self,
        ex: BentoMLException,
        context: BentoServicerContext,
        method_name: str,
    ) -> None:
        """Handle an exception raised by a method.

        Args:
            ex: The exception raised by the method.
            request_or_iterator: The request or iterator of responses.
            context: The context of the RPC.
            method_name: The name of the method.
        """
        logger.error(
            f"Error while invoking {method_name}: {ex}", exc_info=sys.exc_info()
        )
        await context.abort(code=grpc_status_code(ex), details=str(ex))

    async def generate_responses(
        self,
        context: BentoServicerContext,
        method_name: str,
        response_iterator: t.AsyncIterable[ResponseType],
    ) -> t.AsyncGenerator[t.Any, None]:
        """Yield all the responses, but check for errors along the way."""
        try:
            async for r in response_iterator:
                yield r
        except BentoMLException as ex:
            await self.handle_exception(ex, context, method_name)

    async def intercept(
        self,
        method: t.Callable[
            [RequestType, BentoServicerContext], t.Awaitable[ResponseType]
        ],
        request: RequestType,
        context: BentoServicerContext,
        method_name: str,
    ) -> t.Any:
        try:
            response_or_iterator = method(request, context)
            if not hasattr(response_or_iterator, "__aiter__"):
                return await response_or_iterator
        except BentoMLException as ex:
            await self.handle_exception(ex, context, method_name)

        return self.generate_responses(context, method_name, response_or_iterator)  # type: ignore (unknown variable warning)
