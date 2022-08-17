from __future__ import annotations

import sys
import asyncio
import logging
from typing import TYPE_CHECKING

import grpc
import anyio

from bentoml.exceptions import BentoMLException
from bentoml.exceptions import UnprocessableEntity
from bentoml.grpc.utils import grpc_status_code
from bentoml._internal.service.service import Service

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from logging import _ExcInfoType as ExcInfoType  # type: ignore (private warning)

    from bentoml.grpc.v1 import service_pb2 as pb
    from bentoml.grpc.v1 import service_pb2_grpc as services
    from bentoml.grpc.types import BentoServicerContext
else:
    from bentoml.grpc.utils import import_generated_stubs

    pb, services = import_generated_stubs()


def log_exception(request: pb.Request, exc_info: ExcInfoType) -> None:
    # gRPC will always send a POST request.
    logger.error(f"Exception on /{request.api_name} [POST]", exc_info=exc_info)


def create_bentoservicer(service: Service) -> services.BentoServiceServicer:
    """
    This is the actual implementation of BentoServicer.
    Main inference entrypoint will be invoked via /bentoml.grpc.<version>.BentoService/Call
    """

    class BentoServiceServicer(services.BentoServiceServicer):
        """An asyncio implementation of BentoService servicer."""

        async def Call(  # type: ignore (no async types) # pylint: disable=invalid-overridden-method
            self,
            request: pb.Request,
            context: BentoServicerContext,
        ) -> pb.Response | None:
            if request.api_name not in service.apis:
                raise UnprocessableEntity(
                    f"given 'api_name' is not defined in {service.name}",
                )

            api = service.apis[request.api_name]
            response = pb.Response()

            try:
                input_ = await api.input.from_grpc_request(request, context)

                if asyncio.iscoroutinefunction(api.func):
                    output = await api.func(input_)
                else:
                    output = await anyio.to_thread.run_sync(api.func, input_)

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
            except Exception:  # pylint: disable=broad-except
                log_exception(request, sys.exc_info())
                await context.abort(
                    code=grpc.StatusCode.UNKNOWN,
                    details="An error has occurred in BentoML user code when handling this request, find the error details in server logs.",
                )
            return response

    return BentoServiceServicer()
