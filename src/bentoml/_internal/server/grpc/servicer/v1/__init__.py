from __future__ import annotations

import sys
import typing as t
import asyncio
import logging
from typing import TYPE_CHECKING

import anyio

from .....utils import LazyLoader
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
    from google.protobuf import struct_pb2

    from ......grpc.v1 import service_pb2 as pb
    from ......grpc.v1 import service_pb2_grpc as services
    from ......grpc.types import BentoServicerContext
    from .....service.service import Service
else:
    grpc, _ = import_grpc()
    pb, services = import_generated_stubs(version="v1")
    struct_pb2 = LazyLoader("struct_pb2", globals(), "google.protobuf.struct_pb2")


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

            # NOTE: since IODescriptor.proto_fields is a tuple, the order is preserved.
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
                    if api.multi_input:
                        output = await api.func(**input_data)
                    else:
                        output = await api.func(input_data)
                else:
                    if api.multi_input:
                        output = await anyio.to_thread.run_sync(api.func, **input_data)
                    else:
                        output = await anyio.to_thread.run_sync(api.func, input_data)

                res = await api.output.to_proto(output)
                # TODO(aarnphm): support multiple proto fields
                response = pb.Response(**{api.output.proto_fields[0]: res})
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

        async def ServiceMetadata(  # type: ignore (no async types) # pylint: disable=invalid-overridden-method
            self,
            request: pb.ServiceMetadataRequest,  # pylint: disable=unused-argument
            context: BentoServicerContext,  # pylint: disable=unused-argument
        ) -> pb.ServiceMetadataResponse:
            return pb.ServiceMetadataResponse(
                name=service.name,
                docs=service.doc,
                apis=[
                    pb.ServiceMetadataResponse.InferenceAPI(
                        name=api.name,
                        docs=api.doc,
                        input=make_descriptor_spec(
                            api.input.to_spec(), pb.ServiceMetadataResponse
                        ),
                        output=make_descriptor_spec(
                            api.output.to_spec(), pb.ServiceMetadataResponse
                        ),
                    )
                    for api in service.apis.values()
                ],
            )

    return BentoServiceImpl()


if TYPE_CHECKING:
    NestedDictStrAny = dict[str, dict[str, t.Any] | t.Any]
    TupleAny = tuple[t.Any, ...]


def _tuple_converter(d: NestedDictStrAny | None) -> NestedDictStrAny | None:
    # handles case for struct_pb2.Value where nested items are tuple.
    # if that is the case, then convert to list.
    # This dict is only one level deep, as we don't allow nested Multipart.
    if d is not None:
        for key, value in d.items():
            if isinstance(value, tuple):
                d[key] = list(t.cast("TupleAny", value))
            elif isinstance(value, dict):
                d[key] = _tuple_converter(t.cast("NestedDictStrAny", value))
    return d


def make_descriptor_spec(
    spec: dict[str, t.Any] | None, pb: type[pb.ServiceMetadataResponse]
) -> pb.ServiceMetadataResponse.DescriptorMetadata:
    from .....io_descriptors.json import parse_dict_to_proto

    if spec is not None:
        descriptor_id = spec.pop("id")
        return pb.DescriptorMetadata(
            descriptor_id=descriptor_id,
            attributes=struct_pb2.Struct(
                fields={
                    key: parse_dict_to_proto(
                        _tuple_converter(value), struct_pb2.Value()
                    )
                    for key, value in spec.items()
                }
            ),
        )
    return pb.DescriptorMetadata()
