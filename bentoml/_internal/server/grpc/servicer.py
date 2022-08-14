from __future__ import annotations

import sys
import asyncio
import logging
from typing import TYPE_CHECKING

import grpc
import anyio
from grpc import aio

from bentoml.exceptions import BentoMLException
from bentoml.exceptions import UnprocessableEntity
from bentoml.exceptions import MissingDependencyException
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
    logger.error(f"Exception on /{request.api_name}", exc_info=exc_info)


def register_bento_servicer(service: Service, server: aio.Server) -> None:
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

    _service_pb2_grpc.add_BentoServiceServicer_to_server(BentoServiceServicer(), server)  # type: ignore (lack of asyncio types)


async def register_health_servicer(server: aio.Server) -> None:
    from bentoml.grpc.v1 import service_pb2

    try:
        from grpc_health.v1 import health
        from grpc_health.v1 import health_pb2
        from grpc_health.v1 import health_pb2_grpc
    except ImportError:
        raise MissingDependencyException(
            "'grpcio-health-checking' is required for using health checking endpoints. Install with `pip install grpcio-health-checking`."
        )
    try:
        # reflection is required for health checking to work.
        from grpc_reflection.v1alpha import reflection
    except ImportError:
        raise MissingDependencyException(
            "reflection is enabled, which requires 'grpcio-reflection' to be installed. Install with `pip install 'grpcio-relfection'.`"
        )

    # Create a health check servicer. We use the non-blocking implementation
    # to avoid thread starvation.
    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    # create a list of service we want to export for health checking.
    services = tuple(
        service.full_name
        for service in service_pb2.DESCRIPTOR.services_by_name.values()
    ) + (health.SERVICE_NAME, reflection.SERVICE_NAME)
    reflection.enable_server_reflection(services, server)

    # mark all services as healthy
    for service in services:
        await health_servicer.set(service, health_pb2.HealthCheckResponse.SERVING)  # type: ignore (unfinished grpcio-health-checking type)
