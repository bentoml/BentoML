from __future__ import annotations

import sys
import typing as t
import asyncio
import logging
from abc import ABCMeta
from abc import abstractmethod
from typing import TYPE_CHECKING

import grpc
from grpc import aio

from bentoml.exceptions import BentoMLException

from ....utils.grpc import get_rpc_handler
from ....utils.grpc import grpc_status_code
from ....utils.grpc import invoke_handler_factory

if TYPE_CHECKING:
    from ..types import Request
    from ..types import Response
    from ..types import HandlerMethod
    from ..types import RpcMethodHandler
    from ..types import HandlerCallDetails
    from ..types import BentoServicerContext

AsyncClientInterceptorReturn = type(
    "AsyncClientInterceptorReturn", (aio.Call, grpc.Future), {}
)

logger = logging.getLogger(__name__)


# Modified from https://github.com/d5h-foss/grpc-interceptor
# with addition of better typing that fits with BentoService signatures.
class AsyncServerInterceptor(aio.ServerInterceptor, metaclass=ABCMeta):
    """
    Base class for BentoService server-side interceptors.

    To implement, subclass this class and override ``intercept`` method.
    """

    @abstractmethod
    async def intercept(
        self,
        method: HandlerMethod[t.Any],
        request: Request,
        context: BentoServicerContext,
        method_name: str,
    ) -> t.Any:
        response_or_iterator = method(request, context)
        if hasattr(response_or_iterator, "__aiter__"):
            return response_or_iterator
        else:
            return await response_or_iterator

    async def intercept_service(
        self,
        continuation: t.Callable[[HandlerCallDetails], t.Awaitable[RpcMethodHandler]],
        handler_call_details: HandlerCallDetails,
    ) -> RpcMethodHandler:
        """
        Implementation of grpc.aio.ServerInterceptor.
        Don't override unless you know what you are doing.
        """
        handler = await continuation(handler_call_details)
        handler_factory, next_handler = get_rpc_handler(handler)
        method_name = handler_call_details.method

        if handler.response_streaming:

            async def invoke_intercept_streaming(
                request: Request, context: BentoServicerContext
            ) -> t.AsyncGenerator[Response, None]:
                coroutine_or_asyncgen = await self.intercept(
                    next_handler, request, context, method_name
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
                    asyncgen = await coroutine_or_asyncgen
                    # If a handler is using the read/write API, it will return None.
                    if not asyncgen:
                        return
                else:
                    asyncgen = coroutine_or_asyncgen

                async for r in asyncgen:
                    yield r

            return invoke_handler_factory(
                invoke_intercept_streaming, handler_factory, handler
            )
        else:

            async def invoke_intercept_unary(
                request: Request, context: BentoServicerContext
            ) -> t.Awaitable[Response]:
                return await self.intercept(next_handler, request, context, method_name)

            return invoke_handler_factory(
                invoke_intercept_unary, handler_factory, handler
            )


class ExceptionHandlerInterceptor(AsyncServerInterceptor):
    """An async interceptor that handles exceptions raised via BentoService."""

    async def handle_exception(
        self,
        ex: Exception,
        context: BentoServicerContext,
        method_name: str,
    ) -> None:
        """Handle an exception raised by a method.

        Args:
            ex: The exception raised by the method.
            context: The context of the RPC.
            method_name: The name of the method.
        """
        logger.error(
            f"Error while invoking {method_name}: {ex}", exc_info=sys.exc_info()
        )
        details = f"{ex.__class__.__name__}<{ex}>"
        if isinstance(ex, BentoMLException):
            status_code = grpc_status_code(ex)
            details = ex.message
        elif any(isinstance(ex, cls) for cls in (RuntimeError, TypeError)):
            status_code = grpc.StatusCode.INTERNAL
        else:
            status_code = grpc.StatusCode.UNKNOWN

        await context.abort(code=status_code, details=details)
        raise ex

    async def generate_responses(
        self,
        context: BentoServicerContext,
        method_name: str,
        response_iterator: t.AsyncIterable[Response],
    ) -> t.AsyncGenerator[t.Any, None]:
        """Yield all the responses, but check for errors along the way."""
        try:
            async for r in response_iterator:
                yield r
        except Exception as ex:
            await self.handle_exception(ex, context, method_name)

    async def intercept(
        self,
        method: HandlerMethod[t.Any],
        request: Request,
        context: BentoServicerContext,
        method_name: str,
    ) -> t.Any:
        try:
            response_or_iterator = method(request, context)
            if not hasattr(response_or_iterator, "__aiter__"):
                return await response_or_iterator
        except Exception as ex:
            await self.handle_exception(ex, context, method_name)

        return self.generate_responses(context, method_name, response_or_iterator)  # type: ignore (unknown variable warning)
