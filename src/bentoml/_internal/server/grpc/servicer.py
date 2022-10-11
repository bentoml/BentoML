from __future__ import annotations

import sys
import typing as t
import asyncio
import logging
from typing import TYPE_CHECKING

import anyio

from bentoml.grpc.utils import grpc_status_code
from bentoml.grpc.utils import validate_proto_fields

from ....exceptions import InvalidArgument
from ....exceptions import BentoMLException

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from logging import _ExcInfoType as ExcInfoType  # type: ignore (private warning)

    import grpc
    from grpc import aio
    from grpc_health.v1 import health
    from typing_extensions import Self

    from bentoml.grpc.types import Interceptors
    from bentoml.grpc.types import AddServicerFn
    from bentoml.grpc.types import ServicerClass
    from bentoml.grpc.types import BentoServicerContext
    from bentoml.grpc.v1alpha1 import service_pb2 as pb
    from bentoml.grpc.v1alpha1 import service_pb2_grpc as services

    from ...service.service import Service

else:
    from bentoml.grpc.utils import import_grpc
    from bentoml.grpc.utils import import_generated_stubs

    from ...utils import LazyLoader

    pb, services = import_generated_stubs()
    grpc, aio = import_grpc()
    health = LazyLoader(
        "health",
        globals(),
        "grpc_health.v1.health",
        exc_msg="'grpcio-health-checking' is required for using health checking endpoints. Install with 'pip install grpcio-health-checking'.",
    )
    containers = LazyLoader(
        "containers", globals(), "google.protobuf.internal.containers"
    )


def log_exception(request: pb.Request, exc_info: ExcInfoType) -> None:
    # gRPC will always send a POST request.
    logger.error("Exception on /%s [POST]", request.api_name, exc_info=exc_info)


class Servicer:
    """Create an instance of gRPC Servicer."""

    def __init__(
        self: Self,
        service: Service,
        on_startup: t.Sequence[t.Callable[[], t.Any]] | None = None,
        on_shutdown: t.Sequence[t.Callable[[], t.Any]] | None = None,
        mount_servicers: t.Sequence[tuple[ServicerClass, AddServicerFn, list[str]]]
        | None = None,
        interceptors: Interceptors | None = None,
    ) -> None:
        self.bento_service = service

        self.on_startup = [] if not on_startup else list(on_startup)
        self.on_shutdown = [] if not on_shutdown else list(on_shutdown)
        self.mount_servicers = [] if not mount_servicers else list(mount_servicers)
        self.interceptors = [] if not interceptors else list(interceptors)
        self.loaded = False

    def load(self):
        assert not self.loaded

        self.interceptors_stack = self.build_interceptors_stack()

        self.bento_servicer = create_bento_servicer(self.bento_service)

        # Create a health check servicer. We use the non-blocking implementation
        # to avoid thread starvation.
        self.health_servicer = health.aio.HealthServicer()

        self.service_names = tuple(
            service.full_name for service in pb.DESCRIPTOR.services_by_name.values()
        ) + (health.SERVICE_NAME,)
        self.loaded = True

    def build_interceptors_stack(self) -> list[aio.ServerInterceptor]:
        return list(map(lambda x: x(), self.interceptors))

    async def startup(self):
        for handler in self.on_startup:
            if is_async_iterable(handler):
                await handler()
            else:
                handler()

    async def shutdown(self):
        for handler in self.on_shutdown:
            if is_async_iterable(handler):
                await handler()
            else:
                handler()

    def __bool__(self):
        return self.loaded


def is_async_iterable(obj: t.Any) -> bool:  # pragma: no cover
    return asyncio.iscoroutinefunction(obj) or (
        callable(obj) and asyncio.iscoroutinefunction(obj.__call__)
    )


def create_bento_servicer(service: Service) -> services.BentoServiceServicer:
    """
    This is the actual implementation of BentoServicer.
    Main inference entrypoint will be invoked via /bentoml.grpc.<version>.BentoService/Call
    """
    from ...io_descriptors.multipart import Multipart

    class BentoServiceImpl(services.BentoServiceServicer):
        """An asyncio implementation of BentoService servicer."""

        async def Call(  # type: ignore (no async types) # pylint: disable=invalid-overridden-method
            self,
            request: pb.Request,
            context: BentoServicerContext,
        ) -> pb.Response | None:
            if request.api_name not in service.apis:
                raise InvalidArgument(
                    f"given 'api_name' is not defined in {service.name}",
                ) from None

            api = service.apis[request.api_name]
            response = pb.Response()

            # NOTE: since IODescriptor._proto_fields is a tuple, the order is preserved.
            # This is important so that we know the order of fields to process.
            # We will use fields descriptor to determine how to process that request.
            try:
                # we will check if the given fields list contains a pb.Multipart.
                input_proto = getattr(
                    request,
                    validate_proto_fields(request.WhichOneof("content"), api.input),
                )
                input_data = await api.input.from_proto(input_proto)
                if asyncio.iscoroutinefunction(api.func):
                    if isinstance(api.input, Multipart):
                        output = await api.func(**input_data)
                    else:
                        output = await api.func(input_data)
                else:
                    if isinstance(api.input, Multipart):
                        output = await anyio.to_thread.run_sync(api.func, **input_data)
                    else:
                        output = await anyio.to_thread.run_sync(api.func, input_data)
                res = await api.output.to_proto(output)
                # TODO(aarnphm): support multiple proto fields
                response = pb.Response(**{api.output._proto_fields[0]: res})
            except BentoMLException as e:
                log_exception(request, sys.exc_info())
                await context.abort(code=grpc_status_code(e), details=e.message)
            except (RuntimeError, TypeError, NotImplementedError):
                log_exception(request, sys.exc_info())
                await context.abort(
                    code=grpc.StatusCode.INTERNAL,
                    details="A runtime error has occurred, see stacktrace from logs.",
                )
            except Exception:  # pylint: disable=broad-except
                log_exception(request, sys.exc_info())
                await context.abort(
                    code=grpc.StatusCode.INTERNAL,
                    details="An error has occurred in BentoML user code when handling this request, find the error details in server logs.",
                )
            return response

    return BentoServiceImpl()
