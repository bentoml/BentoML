from __future__ import annotations

import typing as t
import asyncio
import inspect
import logging
import functools
import contextlib
from enum import Enum
from typing import TYPE_CHECKING

from packaging.version import parse

from . import Client
from . import ClientCredentials
from .. import io
from .. import Service
from ..exceptions import BentoMLException
from ..grpc.utils import import_grpc
from ..grpc.utils import parse_method_name
from ..grpc.utils import import_generated_stubs
from ..grpc.utils import LATEST_PROTOCOL_VERSION
from .._internal.utils import LazyLoader
from .._internal.utils import cached_property
from .._internal.server.grpc_app import load_from_file
from .._internal.service.inference_api import InferenceAPI

logger = logging.getLogger(__name__)

PROTOBUF_EXC_MESSAGE = "'protobuf' is required to use gRPC Client. Install with 'pip install bentoml[grpc]'."
REFLECTION_EXC_MESSAGE = "'grpcio-reflection' is required to use gRPC Client. Install with 'pip install bentoml[grpc-reflection]'."

if TYPE_CHECKING:
    from types import TracebackType
    from urllib.parse import ParseResult

    import grpc
    from grpc import aio
    from google.protobuf import message as _message
    from google.protobuf import json_format as _json_format
    from google.protobuf import descriptor_pb2 as pb_descriptor
    from google.protobuf import descriptor_pool as _descriptor_pool
    from google.protobuf import symbol_database as _symbol_database
    from grpc_reflection.v1alpha import reflection_pb2 as pb_reflection
    from grpc_reflection.v1alpha import reflection_pb2_grpc as services_reflection

    # type hint specific imports.
    from google.protobuf.descriptor import MethodDescriptor
    from google.protobuf.descriptor import ServiceDescriptor
    from google.protobuf.descriptor_pb2 import FileDescriptorProto
    from google.protobuf.descriptor_pb2 import MethodDescriptorProto
    from google.protobuf.descriptor_pool import DescriptorPool
    from google.protobuf.symbol_database import SymbolDatabase
    from grpc_reflection.v1alpha.reflection_pb2 import ServiceResponse
    from grpc_reflection.v1alpha.reflection_pb2_grpc import ServerReflectionStub

    from ..grpc.types import MultiCallable

    from ..grpc.v1.service_pb2 import ServiceMetadataResponse
else:
    pb_descriptor = LazyLoader(
        "pb_descriptor",
        globals(),
        "google.protobuf.descriptor_pb2",
        exc_msg=PROTOBUF_EXC_MESSAGE,
    )
    _descriptor_pool = LazyLoader(
        "_descriptor_pool",
        globals(),
        "google.protobuf.descriptor_pool",
        exc_msg=PROTOBUF_EXC_MESSAGE,
    )
    _symbol_database = LazyLoader(
        "_symbol_database",
        globals(),
        "google.protobuf.symbol_database",
        exc_msg=PROTOBUF_EXC_MESSAGE,
    )
    _json_format = LazyLoader(
        "_json_format",
        globals(),
        "google.protobuf.json_format",
        exc_msg=PROTOBUF_EXC_MESSAGE,
    )
    services_reflection = LazyLoader(
        "services_reflection",
        globals(),
        "grpc_reflection.v1alpha.reflection_pb2_grpc",
        exc_msg=REFLECTION_EXC_MESSAGE,
    )
    pb_reflection = LazyLoader(
        "pb_reflection",
        globals(),
        "grpc_reflection.v1alpha.reflection_pb2",
        exc_msg=REFLECTION_EXC_MESSAGE,
    )
    grpc, aio = import_grpc()

_object_setattr = object.__setattr__

if TYPE_CHECKING:

    class RpcMethod(t.TypedDict):
        request_streaming: t.Literal[True, False]
        response_streaming: bool
        input_type: type[t.Any]
        output_type: t.NotRequired[type[t.Any]]
        handler: MultiCallable

else:
    RpcMethod = dict

# TODO: xDS support
class GrpcClient(Client):
    def __init__(
        self,
        server_url: str,
        svc: Service | None = None,
        # gRPC specific options
        ssl: bool = False,
        channel_options: aio.ChannelArgumentType | None = None,
        interceptors: t.Sequence[aio.ClientInterceptor] | None = None,
        compression: grpc.Compression | None = None,
        ssl_client_credentials: ClientCredentials | None = None,
        *,
        protocol_version: str = LATEST_PROTOCOL_VERSION,
    ):
        super().__init__(svc, server_url)

        # Call requires an api_name, therefore we need a reserved keyset of self._svc.apis
        self._rev_apis = {v: k for k, v in self._svc.apis.items()}

        self._protocol_version = protocol_version
        self._compression = compression
        self._options = channel_options
        self._interceptors = interceptors
        self._channel = None
        self._credentials = None
        if ssl:
            assert (
                ssl_client_credentials is not None
            ), "'ssl=True' requires 'credentials'"
            self._credentials = grpc.ssl_channel_credentials(
                **{
                    k: load_from_file(v) if isinstance(v, str) else v
                    for k, v in ssl_client_credentials.items()
                }
            )

        self._descriptor_pool: DescriptorPool = _descriptor_pool.Default()
        self._symbol_database: SymbolDatabase = _symbol_database.Default()

        self._available_services: tuple[str, ...] = tuple()
        # cached of all available rpc for a given service.
        self._service_cache: dict[str, dict[str, RpcMethod]] = {}
        # Sets of FileDescriptorProto name to be registered
        self._registered_file_name: set[str] = set()
        self._reflection_stub: ServerReflectionStub | None = None

    @cached_property
    def channel(self):
        if not self._channel:
            if self._credentials is not None:
                self._channel = aio.secure_channel(
                    self.server_url,
                    credentials=self._credentials,
                    options=self._options,
                    compression=self._compression,
                    interceptors=self._interceptors,
                )
            self._channel = aio.insecure_channel(
                self.server_url,
                options=self._options,
                compression=self._compression,
                interceptors=self._interceptors,
            )
        return self._channel

    @staticmethod
    def make_rpc_method(service_name: str, method: str):
        return f"/{service_name}/{method}"

    @property
    def _call_rpc_method(self):
        return self.make_rpc_method(
            f"bentoml.grpc.{self._protocol_version}.BentoService", "Call"
        )

    @cached_property
    def _reserved_kw_mapping(self):
        return {
            "default": f"bentoml.grpc.{self._protocol_version}.BentoService",
            "health": "grpc.health.v1.Health",
            "reflection": "grpc.reflection.v1alpha.ServerReflection",
        }

    async def _exit(self):
        try:
            if self._channel:
                if self._channel.get_state() == grpc.ChannelConnectivity.IDLE:
                    await self._channel.close()
        except AttributeError as e:
            logger.error(f"Error closing channel: %s", e, exc_info=e)
            raise

    def __enter__(self):
        return self.service().__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        try:
            if exc_type is not None:
                self.service().__exit__(exc_type, exc, traceback)
            self._loop.run_until_complete(self._exit())
        except Exception as err:  # pylint: disable=broad-except
            logger.error(f"Exception occurred: %s (%s)", err, exc_type, exc_info=err)
        return False

    @contextlib.contextmanager
    def service(self, service_name: str = "default"):
        stack = contextlib.AsyncExitStack()

        async def close():
            await stack.aclose()

        async def enter():
            res = await stack.enter_async_context(
                self.aservice(service_name, _wrap_in_sync=True)
            )
            return res

        try:
            yield self._loop.run_until_complete(enter())
        finally:
            self._loop.run_until_complete(close())

    async def __aenter__(self):
        return await self.aservice().__aenter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        try:
            if exc_type is not None:
                await self.aservice().__aexit__(exc_type, exc, traceback)
            await self._exit()
        except Exception as err:  # pylint: disable=broad-except
            logger.error(f"Exception occurred: %s (%s)", err, exc_type, exc_info=err)
        return False

    @contextlib.asynccontextmanager
    async def aservice(
        self, service_name: str = "default", *, _wrap_in_sync: bool = False
    ) -> t.AsyncGenerator[t.Self, None]:
        # This is the entrypoint for user to instantiate a client for a given service.

        # default is a special case for BentoService proto.
        if service_name in self._reserved_kw_mapping:
            service_name = self._reserved_kw_mapping[service_name]

        if not self._available_services:
            resp = await self._do_one_request(
                pb_reflection.ServerReflectionRequest(list_services="")
            )
            if resp is not None:
                services: list[ServiceResponse] = resp.list_services_response.service
                self._available_services = tuple(
                    [t.cast(str, s.name) for s in services]
                )

        if (
            service_name in self._available_services
            and service_name not in self._service_cache
        ):
            await self._register_service(service_name)

        if self.channel.get_state() != grpc.ChannelConnectivity.READY:
            # create a blocking call to wait til channel is ready.
            await self.channel.channel_ready()

        try:
            method_meta = self._service_cache[service_name]
        except KeyError:
            raise ValueError(
                f"Failed to find service '{service_name}'. Available: {list(self._service_cache.keys())}"
            ) from None

        def _register(method: str):
            finaliser = f = functools.partial(
                self._invoke,
                self.make_rpc_method(service_name, method),
                _serialize_input=True,
            )
            if _wrap_in_sync:
                # We will have to run the async function in a sync wrapper
                @functools.wraps(f)
                def wrapper(*args: t.Any, **kwargs: t.Any):
                    coro = f(*args, **kwargs)
                    task = asyncio.ensure_future(coro, loop=self._loop)
                    try:
                        res = self._loop.run_until_complete(task)
                        if inspect.isasyncgen(res):
                            # If this is an async generator, then we need to yield again
                            async def call():
                                return await res.__anext__()

                            return self._loop.run_until_complete(call())
                        return res
                    except BaseException:
                        # Consume all exceptions.
                        if task.done() and not task.cancelled():
                            task.exception()
                        raise

                finaliser = wrapper
            _object_setattr(self, method, finaliser)

        # Register all RPC method.
        for method in reversed(method_meta):
            _register(method)

        yield self

    async def _register_service(self, service_name: str) -> None:
        svc_descriptor: ServiceDescriptor | None = None
        try:
            svc_descriptor = self._descriptor_pool.FindServiceByName(service_name)
        except KeyError:
            file_descriptor = await self._find_descriptor_by_symbol(service_name)
            await self._add_file_descriptor(file_descriptor)
            # try to register from FileDescriptorProto again.
            svc_descriptor = self._descriptor_pool.FindServiceByName(service_name)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "Failed to register %s. This might have already been registered.",
                service_name,
                exc_info=e,
            )
            raise
        finally:
            if svc_descriptor is not None:
                self._service_cache[service_name] = self._register_methods(
                    svc_descriptor
                )

    def _register_methods(
        self, service_descriptor: ServiceDescriptor
    ) -> dict[str, RpcMethod]:
        service_descriptor_proto = pb_descriptor.ServiceDescriptorProto()
        service_descriptor.CopyToProto(service_descriptor_proto)
        full_name = service_descriptor.full_name
        metadata: dict[str, RpcMethod] = {}
        for method_proto in service_descriptor_proto.method:
            method_name = method_proto.name
            method_descriptor: MethodDescriptor = service_descriptor.FindMethodByName(
                method_name
            )
            input_type = self._symbol_database.GetPrototype(
                method_descriptor.input_type
            )
            output_type = self._symbol_database.GetPrototype(
                method_descriptor.output_type
            )
            metadata[method_name] = RpcMethod(
                request_streaming=method_proto.client_streaming,
                response_streaming=method_proto.server_streaming,
                input_type=input_type,
                output_type=output_type,
                handler=getattr(
                    self.channel,
                    _RpcType.from_method_descriptor(method_proto),
                )(
                    method=f"/{full_name}/{method_name}",
                    request_serializer=input_type.SerializeToString,
                    response_deserializer=output_type.FromString,
                ),
            )
        return metadata

    async def _add_file_descriptor(self, file_descriptor: FileDescriptorProto):
        dependencies = file_descriptor.dependency
        for deps in dependencies:
            if deps not in self._registered_file_name:
                d_descriptor = await self._find_descriptor_by_filename(deps)
                await self._add_file_descriptor(d_descriptor)
                self._registered_file_name.add(deps)
        self._descriptor_pool.Add(file_descriptor)

    async def _find_descriptor_by_symbol(self, symbol: str):
        req = pb_reflection.ServerReflectionRequest(file_containing_symbol=symbol)
        res = await self._do_one_request(req)
        assert res is not None
        fdp: list[bytes] = res.file_descriptor_response.file_descriptor_proto
        return pb_descriptor.FileDescriptorProto.FromString(fdp[0])

    async def _find_descriptor_by_filename(self, name: str):
        req = pb_reflection.ServerReflectionRequest(file_by_filename=name)
        res = await self._do_one_request(req)
        assert res is not None
        fdp: list[bytes] = res.file_descriptor_response.file_descriptor_proto
        return pb_descriptor.FileDescriptorProto.FromString(fdp[0])

    def _reflection_request(
        self, *reqs: pb_reflection.ServerReflectionRequest
    ) -> t.AsyncIterator[pb_reflection.ServerReflectionResponse]:
        if self._reflection_stub is None:
            # ServerReflectionInfo is a stream RPC, hence the generator.
            self._reflection_stub = services_reflection.ServerReflectionStub(
                self.channel
            )
        return t.cast(
            t.AsyncIterator[pb_reflection.ServerReflectionResponse],
            self._reflection_stub.ServerReflectionInfo((r for r in reqs)),
        )

    async def _do_one_request(
        self, req: pb_reflection.ServerReflectionRequest
    ) -> pb_reflection.ServerReflectionResponse | None:
        try:
            async for r in self._reflection_request(req):
                return r
        except aio.AioRpcError as err:
            code = err.code()
            if code == grpc.StatusCode.UNIMPLEMENTED:
                raise BentoMLException(
                    f"[{code}] Couldn't locate servicer method. The running server might not have reflection enabled. Make sure to pass '--enable-reflection'"
                )
            raise BentoMLException(
                f"Caught AioRpcError while handling reflection request: {err}"
            ) from None

    async def _invoke(
        self,
        method_name: str,
        _serialize_input: bool = False,
        **attrs: t.Any,
    ):
        mn, _ = parse_method_name(method_name)
        if mn.fully_qualified_service not in self._available_services:
            raise ValueError(
                f"{mn.service} is not available in server. Registered services: {self._available_services}"
            )
        # channel kwargs include timeout, metadata, credentials, wait_for_ready and compression
        # to pass it in kwargs add prefix _channel_<args>
        channel_kwargs = {
            k: attrs.pop(f"_channel_{k}", None)
            for k in {
                "timeout",
                "metadata",
                "credentials",
                "wait_for_ready",
                "compression",
            }
        }

        mn, is_valid = parse_method_name(method_name)
        if not is_valid:
            raise ValueError(
                f"{method_name} is not a valid method name. Make sure to follow the format '/package.ServiceName/MethodName'"
            )
        try:
            rpc_method = self._service_cache[mn.fully_qualified_service][mn.method]
        except KeyError:
            raise BentoMLException(
                f"Method '{method_name}' is not registered in current service client."
            ) from None

        handler_type = _RpcType.from_streaming_type(
            rpc_method["request_streaming"], rpc_method["response_streaming"]
        )

        if _serialize_input:
            parsed = handler_type.request_serializer(rpc_method["input_type"], **attrs)
        else:
            parsed = rpc_method["input_type"](**attrs)
        if handler_type.is_unary_response():
            result = await t.cast(
                t.Awaitable[t.Any],
                rpc_method["handler"](parsed, **channel_kwargs),
            )
            return result
        # streaming response
        return handler_type.response_deserializer(
            rpc_method["handler"](parsed, **channel_kwargs)
        )

    def _sync_call(
        self,
        inp: t.Any = None,
        *,
        _bentoml_api: InferenceAPI,
        **kwargs: t.Any,
    ):
        with self:
            return self._loop.run_until_complete(
                self._call(inp, _bentoml_api=_bentoml_api, **kwargs)
            )

    async def _call(
        self,
        inp: t.Any = None,
        *,
        _bentoml_api: InferenceAPI,
        **attrs: t.Any,
    ) -> t.Any:
        async with self:
            fn = functools.partial(
                self._invoke,
                **{
                    f"_channel_{k}": attrs.pop(f"_channel_{k}", None)
                    for k in {
                        "timeout",
                        "metadata",
                        "credentials",
                        "wait_for_ready",
                        "compression",
                    }
                },
            )

            if _bentoml_api.multi_input:
                if inp is not None:
                    raise BentoMLException(
                        f"'{_bentoml_api.name}' takes multiple inputs; all inputs must be passed as keyword arguments."
                    )
                serialized_req = await _bentoml_api.input.to_proto(attrs)
            else:
                serialized_req = await _bentoml_api.input.to_proto(inp)

            # A call includes api_name and given proto_fields
            return await fn(
                self._call_rpc_method,
                **{
                    "api_name": self._rev_apis[_bentoml_api],
                    _bentoml_api.input._proto_fields[0]: serialized_req,
                },
            )

    @staticmethod
    def _create_client(parsed: ParseResult, **kwargs: t.Any) -> GrpcClient:
        server_url = parsed.netloc
        protocol_version = kwargs.get("protocol_version", LATEST_PROTOCOL_VERSION)

        # Since v1, we introduce a ServiceMetadata rpc to retrieve bentoml.Service metadata.
        # This means if user are using client for protocol version v1alpha1,
        # then `client.predict` or `client.classify` won't be available.
        # client.Call will still persist for both protocol version.
        dummy_service: Service | None = None
        if parse(protocol_version) < parse("v1"):
            logger.warning(
                "Using protocol version %s older than v1. This means the client won't have service API functions as attributes. To invoke the RPC endpoint, use 'client.Call()'.",
                protocol_version,
            )
        else:
            pb, _ = import_generated_stubs(protocol_version)

            # create an insecure channel to invoke ServiceMetadata rpc
            with grpc.insecure_channel(server_url) as channel:
                # gRPC sync stub is WIP.
                ServiceMetadata = channel.unary_unary(
                    f"/bentoml.grpc.{protocol_version}.BentoService/ServiceMetadata",
                    request_serializer=pb.ServiceMetadataRequest.SerializeToString,
                    response_deserializer=pb.ServiceMetadataResponse.FromString,
                )
                metadata = t.cast(
                    "ServiceMetadataResponse",
                    ServiceMetadata(pb.ServiceMetadataRequest()),
                )
            dummy_service = Service(metadata.name)

            for api in metadata.apis:
                dummy_service.apis[api.name] = InferenceAPI(
                    None,
                    io.from_spec(
                        {
                            "id": api.input.descriptor_id,
                            "args": _json_format.MessageToDict(api.input.attributes)[
                                "args"
                            ],
                        }
                    ),
                    io.from_spec(
                        {
                            "id": api.output.descriptor_id,
                            "args": _json_format.MessageToDict(api.output.attributes)[
                                "args"
                            ],
                        }
                    ),
                    name=api.name,
                    doc=api.docs,
                )

        return GrpcClient(server_url, dummy_service, **kwargs)

    def __del__(self):
        if self._channel:
            try:
                del self._channel
            except Exception:  # pylint: disable=broad-except
                pass


class _RpcType(Enum):
    UNARY_UNARY = 1
    UNARY_STREAM = 2
    STREAM_UNARY = 3
    STREAM_STREAM = 4

    def is_unary_request(self) -> bool:
        return self.name.lower().startswith("unary_")

    def is_unary_response(self) -> bool:
        return self.name.lower().endswith("_unary")

    @classmethod
    def from_method_descriptor(cls, method_descriptor: MethodDescriptorProto) -> str:
        rpcs = cls.from_streaming_type(
            method_descriptor.client_streaming, method_descriptor.server_streaming
        )
        return rpcs.name.lower()

    @classmethod
    def from_streaming_type(
        cls, client_streaming: bool, server_streaming: bool
    ) -> t.Self:
        if not client_streaming and not server_streaming:
            return cls.UNARY_UNARY
        elif client_streaming and not server_streaming:
            return cls.STREAM_UNARY
        elif not client_streaming and server_streaming:
            return cls.UNARY_STREAM
        else:
            return cls.STREAM_STREAM

    @property
    def request_serializer(self) -> t.Callable[..., t.Any]:
        def _(input_type: type[t.Any], **request_data: t.Any):
            data = request_data or {}
            return _json_format.ParseDict(data, input_type())

        def _it(input_type: type[t.Any], request_data: t.Iterable[t.Any]):
            for data in request_data:
                yield _(input_type, **data)

        return _ if self.is_unary_request() else _it

    @property
    def response_deserializer(self) -> t.Callable[..., t.Any]:
        async def _(response: _message.Message):
            return _json_format.MessageToDict(
                response, preserving_proto_field_name=True
            )

        async def _it(response: t.AsyncIterator[_message.Message]):
            async for r in response:
                yield await _(r)

        return _ if self.is_unary_response() else _it
