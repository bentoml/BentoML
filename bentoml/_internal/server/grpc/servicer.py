from __future__ import annotations

import typing as t
import asyncio
from typing import TYPE_CHECKING

from grpc import aio

from bentoml.exceptions import UnprocessableEntity
from bentoml.exceptions import MissingDependencyException

from .utils import handle_grpc_error

if TYPE_CHECKING:
    from bentoml.grpc.v1.service_pb2_grpc import BentoServiceServicer

    from .utils import BentoServicerContext
    from ..grpc_server import GRPCServer


async def add_health_servicer(
    server: aio.Server,
    *,
    enable_reflection: bool = False,
) -> None:
    from bentoml.grpc.v1 import service_pb2

    try:
        from grpc_health.v1 import health
        from grpc_health.v1 import health_pb2
        from grpc_health.v1 import health_pb2_grpc
    except ImportError:
        raise MissingDependencyException(
            "'grpcio-health-checking' is required for using health checking endpoints. Install with `pip install grpcio-health-checking`."
        )
    # Create a health check servicer. We use the non-blocking implementation
    # to avoid thread starvation.
    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    # create a list of service we want to export for health checking.
    services = tuple(
        service.full_name
        for service in service_pb2.DESCRIPTOR.services_by_name.values()
    ) + (health.SERVICE_NAME,)

    if enable_reflection:
        try:
            from grpc_reflection.v1alpha import reflection
        except ImportError:
            raise MissingDependencyException(
                "reflection is enabled, which requires 'grpcio-reflection' to be installed. Install with `pip install 'grpcio-relfection'.`"
            )

        services += (reflection.SERVICE_NAME,)
        reflection.enable_server_reflection(services, server)

    # mark all services as healthy
    for service in services:
        await health_servicer.set(service, health_pb2.HealthCheckResponse.SERVING)  # type: ignore (unfinished grpcio-health-checking type)


def add_bentoservice_servicer(server_cls: GRPCServer) -> None:
    from bentoml.grpc.v1 import service_pb2 as _service_pb2
    from bentoml.grpc.v1 import service_pb2_grpc as _service_pb2_grpc

    service = server_cls.bento_service

    class BentoServiceServicer(_service_pb2_grpc.BentoServiceServicer):
        """An asyncio implementation of BentoService servicer."""

        @handle_grpc_error
        def ServerLive(
            self,
            request: _service_pb2.ServerLiveRequest,
            context: BentoServicerContext,
        ) -> _service_pb2.ServerLiveResponse:
            return _service_pb2.ServerLiveResponse(live=server_cls.probe.is_live)

        @handle_grpc_error
        def ServerReady(
            self,
            request: _service_pb2.ServerReadyRequest,
            context: BentoServicerContext,
        ) -> _service_pb2.ServerReadyResponse:
            return _service_pb2.ServerReadyResponse(ready=server_cls.probe.is_ready)

        @handle_grpc_error
        async def Call(
            self,
            request: _service_pb2.CallRequest,
            context: BentoServicerContext,
        ) -> _service_pb2.CallResponse | None:
            if request.api_name not in service.apis:
                raise UnprocessableEntity(
                    f"given 'api_name' is not defined in {service.name}",
                )

            api = service.apis[request.api_name]

            input = await api.input.from_grpc_request(request, context)

            if asyncio.iscoroutinefunction(api.func):
                output = await api.func(input)
            else:
                output = api.func(input)

            response = await api.output.to_grpc_response(output, context)
            return response

    _service_pb2_grpc.add_BentoServiceServicer_to_server(
        BentoServiceServicer(), server_cls.server
    )
