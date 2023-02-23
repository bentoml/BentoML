from __future__ import annotations

import sys
import asyncio
import logging
from typing import TYPE_CHECKING

import anyio

from .....context import InferenceApiContext
from ......exceptions import InvalidArgument
from ......exceptions import BentoMLException
from ......grpc.utils import import_grpc
from ......grpc.utils import grpc_status_code
from ......grpc.utils import validate_proto_fields
from ......grpc.utils import import_generated_stubs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from logging import _ExcInfoType as ExcInfoType  # type: ignore (private warning)

    import grpc
    from grpc import aio

    from ......grpc.types import BentoServicerContext
    from ......grpc.v1alpha1 import service_pb2 as pb
    from ......grpc.v1alpha1 import service_pb2_grpc as services
    from .....service.service import Service
else:
    grpc, aio = import_grpc()
    pb, services = import_generated_stubs(version="v1alpha1")


def log_exception(request: pb.Request, exc_info: ExcInfoType) -> None:
    # gRPC will always send a POST request.
    logger.error("Exception on /%s [POST]", request.api_name, exc_info=exc_info)


def create_bento_servicer(service: Service) -> services.BentoServiceServicer:
    """
    This is the actual implementation of BentoServicer.
    Main inference entrypoint will be invoked via /bentoml.grpc.<version>.BentoService/Call
    """

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
            output = None

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
                ctx: InferenceApiContext | None = None
                assert api.func is not None
                if asyncio.iscoroutinefunction(api.func):
                    if api.multi_input:
                        if api.needs_ctx:
                            ctx = InferenceApiContext.from_grpc(context)
                            input_data[api.ctx_param] = ctx
                        output = await api.func(**input_data)
                    else:
                        if api.needs_ctx:
                            ctx = InferenceApiContext.from_grpc(context)
                            output = await api.func(input_data, ctx)
                        else:
                            output = await api.func(input_data)
                else:
                    if api.multi_input:
                        if api.needs_ctx:
                            ctx = InferenceApiContext.from_grpc(context)
                            input_data[api.ctx_param] = ctx
                        output = await anyio.to_thread.run_sync(api.func, **input_data)
                    else:
                        if api.needs_ctx:
                            ctx = InferenceApiContext.from_grpc(context)
                            output = await anyio.to_thread.run_sync(
                                api.func, input_data, ctx
                            )
                        else:
                            output = await anyio.to_thread.run_sync(
                                api.func, input_data
                            )
                # Set gRPC metadata if context is available
                if ctx is not None:
                    context.set_trailing_metadata(
                        aio.Metadata.from_tuple(tuple(ctx.response.metadata.items()))
                    )
                if hasattr(api.output, "to_proto_v1alpha1"):
                    # special case for handling v1alpha1 specific to_proto logic
                    res = await getattr(api.output, "to_proto_v1alpha1")(output)
                else:
                    res = await api.output.to_proto(output)
                # TODO(aarnphm): support multiple proto fields
                response = pb.Response(**{api.output._proto_fields[0]: res})
            except BentoMLException as e:
                log_exception(request, sys.exc_info())
                if output is not None:
                    import inspect

                    signature = inspect.signature(api.output.to_proto)
                    param = next(iter(signature.parameters.values()))
                    ann = ""
                    if param is not inspect.Parameter.empty:
                        ann = param.annotation

                    # more descriptive errors if output is available
                    logger.error(
                        "Function '%s' has 'input=%s,output=%s' as IO descriptor, and returns 'result=%s', while expected return type is '%s'",
                        api.name,
                        api.input,
                        api.output,
                        type(output),
                        ann,
                    )
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
