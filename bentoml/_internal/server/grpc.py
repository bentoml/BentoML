from __future__ import annotations

import sys
import typing as t
import asyncio
from typing import TYPE_CHECKING
from concurrent import futures

if TYPE_CHECKING:
    from grpc_reflection.v1alpha import reflection

    from ..service import Service
else:
    _exc_msg = "'grpcio' is missing. Install with `pip install 'grpcio-reflection'`."
    reflection = LazyLoader(
        "reflection", globals(), "grpc_reflection.v1alpha.reflection", _exc_msg
    )

import grpc
from grpc_health.v1 import health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

from bentoml.grpc.v1 import service_pb2
from bentoml.grpc.v1 import service_pb2_grpc


def get_health_check_servicer(
    server: grpc.aio.Server, max_workers: int = 10
) -> t.Type[health.HealthServicer]:
    # Create a health check servicer. We use the non-blocking implementation
    # to avoid thread starvation.
    health_servicer = health.HealthServicer(
        experimental_non_blocking=True,
        experimental_thread_pool=futures.ThreadPoolExecutor(max_workers=10),
    )


def get_service_servicer(cls: Service) -> t.Type[service_pb2_grpc.BentoServiceServicer]:
    from bentoml.grpc.v1 import service_pb2
    from bentoml.grpc.v1 import service_pb2_grpc

    class BentoServiceServicer(service_pb2_grpc.BentoServiceServicer):
        async def Call(
            self,
            request: service_pb2.Request,
            context: grpc.aio.ServicerContext[
                service_pb2.Request, service_pb2.Response
            ],
        ) -> service_pb2.Response:
            if request.api_name not in cls.apis:
                await context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    f"given 'api_name' is not defined in {cls.name}",
                )
                return service_pb2.Response()

            api = cls.apis[request.api_name]

            try:
                input = await api.input.from_grpc_request(request, context)

                if asyncio.iscoroutinefunction(api.func):
                    output = await api.func(input)
                else:
                    output = api.func(input)

                response = await api.output.to_grpc_response(output, context)
            except grpc.aio.AioRpcError:
                await context.abort(
                    grpc.StatusCode.INTERNAL,
                    f"Error while calling RPC: {sys.exc_info()}",
                )
                response = service_pb2.Response()
            except Exception:
                await context.abort(grpc.StatusCode.UNKNOWN, str(sys.exc_info()))
                response = service_pb2.Response()
            return response

    return BentoServiceServicer
