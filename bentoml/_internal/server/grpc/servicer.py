from __future__ import annotations

import sys
import asyncio
import logging
from typing import TYPE_CHECKING

import grpc
import anyio

from bentoml.exceptions import BentoMLException
from bentoml.exceptions import UnprocessableEntity
from bentoml._internal.service.service import Service

from ...utils import LazyLoader
from ...utils.grpc import grpc_status_code

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from logging import _ExcInfoType as ExcInfoType  # type: ignore (private warning)

    from bentoml.grpc.v1 import service_pb2 as _service_pb2
    from bentoml.grpc.v1 import service_pb2_grpc as _service_pb2_grpc

    from .types import BentoServicerContext
else:
    _service_pb2 = LazyLoader("_service_pb2", globals(), "bentoml.grpc.v1.service_pb2")
    _service_pb2_grpc = LazyLoader(
        "_service_pb2_grpc", globals(), "bentoml.grpc.v1.service_pb2_grpc"
    )


def log_exception(request: _service_pb2.Request, exc_info: ExcInfoType) -> None:
    # gRPC will always send a POST request.
    logger.error(f"Exception on /{request.api_name} [POST]", exc_info=exc_info)


def create_bento_servicer(service: Service) -> _service_pb2_grpc.BentoServiceServicer:
    """
    This is the actual implementation of BentoServicer.
    Main inference entrypoint will be invoked via /bentoml.grpc.<version>.BentoService/Call
    """

    class BentoServiceServicer(_service_pb2_grpc.BentoServiceServicer):
        """An asyncio implementation of BentoService servicer."""

        async def Call(  # type: ignore (no async types)
            self,
            request: _service_pb2.Request,
            context: BentoServicerContext,
        ) -> _service_pb2.Response | None:
            if request.api_name not in service.apis:
                raise UnprocessableEntity(
                    f"given 'api_name' is not defined in {service.name}",
                )

            api = service.apis[request.api_name]
            response = _service_pb2.Response()

            try:
                input = await api.input.from_grpc_request(request, context)

                if asyncio.iscoroutinefunction(api.func):
                    output = await api.func(input)
                else:
                    output = await anyio.to_thread.run_sync(api.func, input)

                response = await api.output.to_grpc_response(output, context)
            except BentoMLException as e:
                log_exception(request, sys.exc_info())
                await context.abort(code=grpc_status_code(e), details=e.message)
            except (RuntimeError, TypeError, NotImplementedError):
                log_exception(request, sys.exc_info())
                await context.abort(
                    code=grpc.StatusCode.INTERNAL,
                    details="An internal runtime error has occurred, check out error details in server logs.",
                )
            except Exception:  # type: ignore (generic exception)
                log_exception(request, sys.exc_info())
                await context.abort(
                    code=grpc.StatusCode.UNKNOWN,
                    details="An error has occurred in BentoML user code when handling this request, find the error details in server logs.",
                )
            return response

    return BentoServiceServicer()
