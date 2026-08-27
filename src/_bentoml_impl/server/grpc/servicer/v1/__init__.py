from __future__ import annotations

import functools
import inspect
import logging
import sys
import typing as t
from typing import TYPE_CHECKING

import anyio

from _bentoml_impl.grpc.codec import decode_proto
from _bentoml_impl.grpc.codec import encode_proto
from _bentoml_sdk.io_models import ARGS
from _bentoml_sdk.io_models import KWARGS
from _bentoml_sdk.io_models import IORootModel
from bentoml._internal.utils import get_original_func
from bentoml._internal.utils import is_async_callable
from bentoml._internal.utils.lazy_loader import LazyLoader
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import InvalidArgument
from bentoml.grpc.utils import grpc_status_code
from bentoml.grpc.utils import import_generated_stubs
from bentoml.grpc.utils import import_grpc

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from logging import _ExcInfoType as ExcInfoType  # type: ignore (private warning)

    import grpc
    from google.protobuf import struct_pb2
    from grpc import aio
    from starlette.requests import Request

    from _bentoml_sdk import Service
    from bentoml.grpc.types import BentoServicerContext
    from bentoml.grpc.v1 import service_pb2 as pb
    from bentoml.grpc.v1 import service_pb2_grpc as services
else:
    grpc, aio = import_grpc()
    pb, services = import_generated_stubs(version="v1")
    struct_pb2 = LazyLoader("struct_pb2", globals(), "google.protobuf.struct_pb2")


def log_exception(request: pb.Request, exc_info: ExcInfoType) -> None:
    logger.error("Exception on /%s [POST]", request.api_name, exc_info=exc_info)


def _metadata_item(item: t.Any) -> tuple[t.Any, t.Any]:
    if hasattr(item, "key") and hasattr(item, "value"):
        return item.key, item.value
    return item


def _metadata_bytes(value: t.Any) -> bytes:
    if isinstance(value, bytes):
        return value
    return str(value).encode("latin-1")


def _request_from_grpc(request: pb.Request, context: BentoServicerContext) -> Request:
    from starlette.requests import Request

    invocation_metadata = getattr(context, "invocation_metadata", None)
    metadata = invocation_metadata() if invocation_metadata is not None else None
    headers = [
        (_metadata_bytes(key), _metadata_bytes(value))
        for key, value in map(_metadata_item, metadata or ())
    ]

    async def receive() -> dict[str, t.Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "2",
            "method": "POST",
            "scheme": "http",
            "path": f"/{request.api_name}",
            "raw_path": f"/{request.api_name}".encode(),
            "query_string": b"",
            "headers": headers,
            "client": None,
            "server": None,
            "root_path": "",
            "state": {},
        },
        receive,
    )


def _propagate_response_metadata(ctx: t.Any, context: BentoServicerContext) -> None:
    raw_metadata = getattr(ctx.response.metadata, "raw", ())
    if not raw_metadata:
        return

    outgoing = tuple(
        (key.decode("latin-1"), value.decode("latin-1"))
        for key, value in raw_metadata
    )
    trailing_metadata = getattr(context, "trailing_metadata", None)
    existing = trailing_metadata() if trailing_metadata is not None else None
    existing_items = tuple(map(_metadata_item, existing or ()))
    context.set_trailing_metadata((*existing_items, *outgoing))


def _call_args_from_input(method: t.Any, input_data: t.Any, ctx: t.Any) -> tuple[
    tuple[t.Any, ...], dict[str, t.Any]
]:
    call_args: tuple[t.Any, ...] = ()
    call_kwargs: dict[str, t.Any] = {}
    if getattr(method.input_spec, "__root_input__", False):
        if isinstance(input_data, IORootModel):
            call_args = (input_data.root,)
        else:
            call_args = (input_data,)
    else:
        call_kwargs = {k: getattr(input_data, k) for k in input_data.model_fields}
    if method.ctx_param is not None:
        call_kwargs[method.ctx_param] = ctx
    if ARGS in call_kwargs:
        call_args = (*call_args, call_kwargs.pop(ARGS))
    if KWARGS in call_kwargs:
        call_kwargs.update(call_kwargs.pop(KWARGS))
    return call_args, call_kwargs


def create_bento_servicer(service: Service[t.Any]) -> services.BentoServiceServicer:
    """Create a v1 BentoService servicer for a new-style ``@bentoml.service()``."""

    class BentoServiceImpl(services.BentoServiceServicer):
        instance: t.Any = None

        def set_instance(self, instance: t.Any) -> None:
            self.instance = instance

        def _get_instance(self) -> t.Any:
            if self.instance is None:
                self.instance = service()
            return self.instance

        async def Call(  # type: ignore (no async types)
            self,
            request: pb.Request,
            context: BentoServicerContext,
        ) -> pb.Response | None:
            response = pb.Response()
            try:
                grpc_request = _request_from_grpc(request, context)
                with service.context.in_request(grpc_request) as ctx:
                    if request.api_name not in service.apis:
                        raise InvalidArgument(
                            f"given 'api_name' is not defined in {service.name}",
                        ) from None

                    method = service.apis[request.api_name]
                    if method.is_stream:
                        await context.abort(
                            code=grpc.StatusCode.UNIMPLEMENTED,
                            details=(
                                f"API {method.name!r} is a streaming endpoint; "
                                "gRPC streaming is not supported for @bentoml.service() yet"
                            ),
                        )
                        return None
                    if method.batchable:
                        await context.abort(
                            code=grpc.StatusCode.UNIMPLEMENTED,
                            details=(
                                f"API {method.name!r} is batchable; "
                                "adaptive batching is not supported over gRPC for @bentoml.service() yet"
                            ),
                        )
                        return None
                    if method.is_task:
                        await context.abort(
                            code=grpc.StatusCode.UNIMPLEMENTED,
                            details=(
                                f"API {method.name!r} is a task endpoint; "
                                "tasks are not supported over gRPC for @bentoml.service() yet"
                            ),
                        )
                        return None

                    field = request.WhichOneof("content")
                    input_data = await decode_proto(
                        method.input_spec,
                        field,
                        getattr(request, field) if field else None,
                    )
                    call_args, call_kwargs = _call_args_from_input(
                        method, input_data, ctx
                    )
                    func = getattr(self._get_instance(), method.name).local
                    original_func = get_original_func(func)
                    if is_async_callable(original_func) or inspect.iscoroutinefunction(
                        original_func
                    ):
                        output = await func(*call_args, **call_kwargs)
                    else:
                        output = await anyio.to_thread.run_sync(
                            functools.partial(func, *call_args, **call_kwargs)
                        )

                    field_name, encoded = await encode_proto(method.output_spec, output)
                    response = pb.Response(**{field_name: encoded})
                    _propagate_response_metadata(ctx, context)
            except BentoMLException as e:
                log_exception(request, sys.exc_info())
                await context.abort(code=grpc_status_code(e), details=e.message)
            except aio.AbortError:
                raise
            except Exception:  # pylint: disable=broad-except
                log_exception(request, sys.exc_info())
                await context.abort(
                    code=grpc.StatusCode.INTERNAL,
                    details="An error has occurred in BentoML user code when handling this request, find the error details in server logs.",
                )
            return response

        async def ServiceMetadata(  # type: ignore (no async types)
            self,
            request: pb.ServiceMetadataRequest,  # pylint: disable=unused-argument
            context: BentoServicerContext,  # pylint: disable=unused-argument
        ) -> pb.ServiceMetadataResponse:
            from google.protobuf.json_format import ParseDict

            def _schema_metadata(schema: dict[str, t.Any]) -> t.Any:
                attributes = struct_pb2.Struct()
                ParseDict(schema, attributes)
                return pb.ServiceMetadataResponse.DescriptorMetadata(
                    descriptor_id="bentoml.sdk.IODescriptor",
                    attributes=attributes,
                )

            return pb.ServiceMetadataResponse(
                name=service.name,
                docs=service.description or "",
                apis=[
                    pb.ServiceMetadataResponse.InferenceAPI(
                        name=api.name,
                        docs=api.doc or "",
                        input=_schema_metadata(api.schema().get("input") or {}),
                        output=_schema_metadata(api.schema().get("output") or {}),
                    )
                    for api in service.apis.values()
                ],
            )

    return BentoServiceImpl()
