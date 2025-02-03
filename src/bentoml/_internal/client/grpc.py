from __future__ import annotations

import asyncio
import logging
import time
import typing as t
from functools import cached_property
from typing import TYPE_CHECKING

from packaging.version import parse

from ...exceptions import BentoMLException
from ...grpc.utils import LATEST_PROTOCOL_VERSION
from ...grpc.utils import import_generated_stubs
from ...grpc.utils import import_grpc
from ...grpc.utils import load_from_file
from .. import io_descriptors
from ..service import Service
from ..service.inference_api import InferenceAPI
from ..utils.lazy_loader import LazyLoader
from . import AsyncClient
from . import Client
from . import SyncClient

logger = logging.getLogger(__name__)

PROTOBUF_EXC_MESSAGE = "'protobuf' is required to use gRPC Client. Install with 'pip install bentoml[grpc]'."
REFLECTION_EXC_MESSAGE = "'grpcio-reflection' is required to use gRPC Client. Install with 'pip install bentoml[grpc-reflection]'."

if TYPE_CHECKING:
    import grpc
    from google.protobuf import json_format as _json_format
    from grpc import aio
    from grpc._channel import Channel as GrpcSyncChannel
    from grpc.aio._channel import Channel as GrpcAsyncChannel
    from grpc_health.v1 import health_pb2 as pb_health

    from ...grpc.v1.service_pb2 import Response
    from ...grpc.v1.service_pb2 import ServiceMetadataResponse
    from ..types import PathType

    class ClientCredentials(t.TypedDict):
        root_certificates: t.NotRequired[PathType | bytes]
        private_key: t.NotRequired[PathType | bytes]
        certificate_chain: t.NotRequired[PathType | bytes]

else:
    grpc, aio = import_grpc()
    pb_health = LazyLoader("pb_health", globals(), "grpc_health.v1.health_pb2")
    _json_format = LazyLoader(
        "_json_format",
        globals(),
        "google.protobuf.json_format",
        exc_msg=PROTOBUF_EXC_MESSAGE,
    )


class GrpcClient(Client):
    def __init__(self, svc: Service, server_url: str):
        self._sync_client = SyncGrpcClient(svc=svc, server_url=server_url)
        self._async_client = AsyncGrpcClient(svc=svc, server_url=server_url)
        super().__init__(svc, server_url)


# TODO: xDS support
class AsyncGrpcClient(AsyncClient):
    def __init__(
        self,
        server_url: str,
        svc: Service,
        # gRPC specific options
        ssl: bool = False,
        channel_options: aio.ChannelArgumentType | None = None,
        interceptors: t.Sequence[aio.ClientInterceptor] | None = None,
        compression: grpc.Compression | None = None,
        ssl_client_credentials: ClientCredentials | None = None,
        *,
        protocol_version: str = LATEST_PROTOCOL_VERSION,
        **kwargs: t.Any,
    ):
        self._pb, _ = import_generated_stubs(protocol_version)

        self._protocol_version = protocol_version
        self._compression = compression
        self._options = channel_options
        self._interceptors = interceptors
        self._credentials = None
        if ssl:
            assert ssl_client_credentials is not None, (
                "'ssl=True' requires 'ssl_client_credentials'"
            )
            self._credentials = grpc.ssl_channel_credentials(
                **{
                    k: load_from_file(v) if isinstance(v, str) else v
                    for k, v in ssl_client_credentials.items()
                }
            )
        self._call_rpc = f"/bentoml.grpc.{protocol_version}.BentoService/Call"
        super().__init__(svc, server_url)

    @cached_property
    def channel(self) -> GrpcAsyncChannel:
        if self._credentials is not None:
            return aio.secure_channel(
                self.server_url,
                credentials=self._credentials,
                options=self._options,
                compression=self._compression,
                interceptors=self._interceptors,
            )
        return aio.insecure_channel(
            self.server_url,
            options=self._options,
            compression=self._compression,
            interceptors=self._interceptors,
        )

    @staticmethod
    def _create_channel(
        server_url: str,
        ssl: bool = False,
        ssl_client_credentials: ClientCredentials | None = None,
        channel_options: t.Any | None = None,
        compression: grpc.Compression | None = None,
    ) -> GrpcAsyncChannel:
        if ssl:
            assert ssl_client_credentials is not None, (
                "'ssl=True' requires 'ssl_client_credentials'"
            )
            return aio.secure_channel(
                server_url,
                credentials=grpc.ssl_channel_credentials(
                    **{
                        k: load_from_file(v) if isinstance(v, str) else v
                        for k, v in ssl_client_credentials.items()
                    }
                ),
                options=channel_options,
                compression=compression,
            )
        return aio.insecure_channel(
            server_url, options=channel_options, compression=compression
        )

    @staticmethod
    async def wait_until_server_ready(
        host: str,
        port: int,
        timeout: float = 30,
        check_interval: int = 1,
        # set kwargs here to omit gRPC kwargs
        **kwargs: t.Any,
    ) -> None:
        protocol_version = kwargs.get("protocol_version", LATEST_PROTOCOL_VERSION)

        async with AsyncGrpcClient._create_channel(
            f"{host}:{port}",
            ssl=kwargs.get("ssl", False),
            ssl_client_credentials=kwargs.get("ssl_client_credentials", None),
            channel_options=kwargs.get("channel_options", None),
            compression=kwargs.get("compression", None),
        ) as channel:
            rpc = channel.unary_unary(
                "/grpc.health.v1.Health/Check",
                request_serializer=pb_health.HealthCheckRequest.SerializeToString,
                response_deserializer=pb_health.HealthCheckResponse.FromString,
            )

            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = t.cast(
                        pb_health.HealthCheckResponse,
                        rpc(
                            pb_health.HealthCheckRequest(
                                service=f"bentoml.grpc.{protocol_version}.BentoService"
                            )
                        ),
                    )
                    if response.status == pb_health.HealthCheckResponse.SERVING:
                        break
                    else:
                        asyncio.sleep(check_interval)
                except grpc.RpcError:
                    logger.debug("Server is not ready. Retrying...")
                    await asyncio.sleep(check_interval)

            try:
                response = t.cast(
                    pb_health.HealthCheckResponse,
                    rpc(
                        pb_health.HealthCheckRequest(
                            service=f"bentoml.grpc.{protocol_version}.BentoService"
                        )
                    ),
                )
                if response.status != pb_health.HealthCheckResponse.SERVING:
                    raise TimeoutError(
                        f"Timed out waiting {timeout} seconds for server at '{host}:{port}' to be ready."
                    )
            except (grpc.RpcError, TimeoutError) as err:
                logger.error("Caught exception while connecting to %s:%s:", host, port)
                logger.error(err)
                raise

    @cached_property
    def _rpc_metadata(self) -> dict[str, dict[str, t.Any]]:
        # Currently all RPCs in BentoService are unary-unary
        # NOTE: we will set the types of the stubs to be Any.
        return {
            method: {"input_type": input_type, "output_type": output_type}
            for method, input_type, output_type in (
                (
                    self._call_rpc,
                    self._pb.Request,
                    self._pb.Response,
                ),
                (
                    f"/bentoml.grpc.{self._protocol_version}.BentoService/ServiceMetadata",
                    self._pb.ServiceMetadataRequest,
                    self._pb.ServiceMetadataResponse,
                ),
                (
                    "/grpc.health.v1.Health/Check",
                    pb_health.HealthCheckRequest,
                    pb_health.HealthCheckResponse,
                ),
            )
        }

    @cached_property
    def _rpc_methods(
        self,
    ) -> dict[str, t.Callable[..., t.Awaitable["Response"]]]:
        def make_async_fn(
            method_name: str,
            input_type: t.Any,
            output_type: t.Any,
        ) -> t.Callable[..., t.Awaitable["Response"]]:
            rpc = self.channel.unary_unary(
                method_name,
                request_serializer=input_type.SerializeToString,
                response_deserializer=output_type.FromString,
            )

            def fn(
                channel_kwargs: t.Dict[str, t.Any],
                method_kwargs: t.Dict[str, t.Any],
            ) -> t.Awaitable["Response"]:
                return t.cast(
                    t.Awaitable["Response"],
                    rpc(input_type(**method_kwargs), **channel_kwargs),
                )

            return fn

        return {
            method_name: make_async_fn(
                method_name,
                input_type=metadata["input_type"],
                output_type=metadata["output_type"],
            )
            for method_name, metadata in self._rpc_metadata.items()
        }

    async def health(self, service_name: str, *, timeout: int = 30) -> t.Any:
        return await self._rpc_methods["/grpc.health.v1.Health/Check"](
            method_kwargs={"service": service_name},
            channel_kwargs={"timeout": timeout},
        )

    @staticmethod
    def _split_channel_args(
        **kwargs: t.Any,
    ) -> tuple[t.Dict[str, t.Any], t.Dict[str, t.Any]]:
        channel_kwarg_names = (
            "timeout",
            "metadata",
            "credentials",
            "wait_for_ready",
            "compression",
        )
        channel_kwargs: t.Dict[str, t.Any] = {}
        other_kwargs: t.Dict[str, t.Any] = {}
        for k, v in kwargs.items():
            if k in channel_kwarg_names:
                channel_kwargs[k] = v
            else:
                other_kwargs[k] = v
        return other_kwargs, channel_kwargs

    async def _call(
        self,
        inp: t.Any = None,
        *,
        _bentoml_api: InferenceAPI[t.Any],
        **attrs: t.Any,
    ) -> t.Any:
        state = self.channel.get_state(try_to_connect=True)
        if state != grpc.ChannelConnectivity.READY:
            # create a blocking call to wait til channel is ready.
            await self.channel.channel_ready()

        if _bentoml_api.multi_input:
            if inp is not None:
                raise BentoMLException(
                    f"'{_bentoml_api.name}' takes multiple inputs; all inputs must be passed as keyword arguments."
                )
            serialized_req = await _bentoml_api.input.to_proto(attrs)
        else:
            serialized_req = await _bentoml_api.input.to_proto(inp)

        # A call includes api_name and given proto_fields
        api_fn = {v: k for k, v in self._svc.apis.items()}
        kwargs, channel_kwargs = self._split_channel_args(**attrs)
        kwargs.update(
            {
                "api_name": api_fn[_bentoml_api],
                _bentoml_api.input.proto_fields[0]: serialized_req,
            },
        )

        if self._call_rpc not in self._rpc_methods:
            raise ValueError(
                f"'{self._call_rpc}' is a yet supported rpc. Current supported are: {self._rpc_metadata}"
            )
        proto = await self._rpc_methods[self._call_rpc](
            channel_kwargs=channel_kwargs,
            method_kwargs=kwargs,
        )
        return await _bentoml_api.output.from_proto(
            getattr(proto, proto.WhichOneof("content"))
        )

    @classmethod
    async def from_url(cls, server_url: str, **kwargs: t.Any) -> AsyncGrpcClient:
        protocol_version = kwargs.get("protocol_version", LATEST_PROTOCOL_VERSION)

        # Since v1, we introduce a ServiceMetadata rpc to retrieve bentoml.Service metadata.
        # then `client.predict` or `client.classify` won't be available.
        # client.Call will still persist for both protocol version.
        if parse(protocol_version) < parse("v1"):
            exception_message = [
                f"Using protocol version {protocol_version} older than v1. 'bentoml.client.Client' will only support protocol version v1 onwards. To create client with protocol version '{protocol_version}', do the following:\n"
                f"""\

from bentoml.grpc.utils import import_generated_stubs, import_grpc

pb, services = import_generated_stubs("{protocol_version}")

grpc, _ = import_grpc()

def run():
    with grpc.insecure_channel("localhost:3000") as channel:
        stubs = services.BentoServiceStub(channel)
        req = stubs.Call(
            request=pb.Request(
                api_name="predict",
                ndarray=pb.NDArray(
                    dtype=pb.NDArray.DTYPE_FLOAT,
                    shape=(1, 4),
                    float_values=[5.9, 3, 5.1, 1.8],
                    ),
                )
            )
            print(req)

if __name__ == '__main__':
    run()
"""
            ]
            raise BentoMLException("\n".join(exception_message))
        pb, _ = import_generated_stubs(protocol_version)

        async with AsyncGrpcClient._create_channel(
            server_url.replace(r"localhost", "0.0.0.0"),
            ssl=kwargs.get("ssl", False),
            ssl_client_credentials=kwargs.get("ssl_client_credentials", None),
            channel_options=kwargs.get("channel_options", None),
            compression=kwargs.get("compression", None),
        ) as channel:
            # create an insecure channel to invoke ServiceMetadata rpc
            metadata = t.cast(
                "ServiceMetadataResponse",
                channel.unary_unary(
                    f"/bentoml.grpc.{protocol_version}.BentoService/ServiceMetadata",
                    request_serializer=pb.ServiceMetadataRequest.SerializeToString,
                    response_deserializer=pb.ServiceMetadataResponse.FromString,
                )(pb.ServiceMetadataRequest()),
            )
        dummy_service = Service(metadata.name)

        for api in metadata.apis:
            try:
                dummy_service.apis[api.name] = InferenceAPI[t.Any](
                    None,
                    io_descriptors.from_spec(
                        {
                            "id": api.input.descriptor_id,
                            "args": _json_format.MessageToDict(
                                api.input.attributes
                            ).get("args", None),
                        }
                    ),
                    io_descriptors.from_spec(
                        {
                            "id": api.output.descriptor_id,
                            "args": _json_format.MessageToDict(
                                api.output.attributes
                            ).get("args", None),
                        }
                    ),
                    name=api.name,
                    doc=api.docs,
                )
            except BentoMLException as e:
                logger.error("Failed to instantiate client for API %s: ", api.name, e)

        return cls(server_url, dummy_service, **kwargs)

    async def close(self):
        await self.channel.close()
        return await super().close()


# TODO: xDS support
class SyncGrpcClient(SyncClient):
    def __init__(
        self,
        server_url: str,
        svc: Service,
        # gRPC specific options
        ssl: bool = False,
        channel_options: aio.ChannelArgumentType | None = None,
        interceptors: t.Sequence[aio.ClientInterceptor] | None = None,
        compression: grpc.Compression | None = None,
        ssl_client_credentials: ClientCredentials | None = None,
        *,
        protocol_version: str = LATEST_PROTOCOL_VERSION,
        **kwargs: t.Any,
    ):
        self._pb, _ = import_generated_stubs(protocol_version)

        self._protocol_version = protocol_version
        self._compression = compression
        self._options = channel_options
        self._interceptors = interceptors
        self._credentials = None
        if ssl:
            assert ssl_client_credentials is not None, (
                "'ssl=True' requires 'ssl_client_credentials'"
            )
            self._credentials = grpc.ssl_channel_credentials(
                **{
                    k: load_from_file(v) if isinstance(v, str) else v
                    for k, v in ssl_client_credentials.items()
                }
            )
        self._call_rpc = f"/bentoml.grpc.{protocol_version}.BentoService/Call"
        super().__init__(svc, server_url)

    @cached_property
    def channel(self) -> GrpcSyncChannel:
        if self._credentials is not None:
            return grpc.secure_channel(
                self.server_url,
                credentials=self._credentials,
                options=self._options,
                compression=self._compression,
            )
        return grpc.insecure_channel(
            self.server_url,
            options=self._options,
            compression=self._compression,
        )

    @staticmethod
    def _create_channel(
        server_url: str,
        ssl: bool = False,
        ssl_client_credentials: ClientCredentials | None = None,
        channel_options: t.Any | None = None,
        compression: grpc.Compression | None = None,
    ) -> GrpcSyncChannel:
        if ssl:
            assert ssl_client_credentials is not None, (
                "'ssl=True' requires 'ssl_client_credentials'"
            )
            return grpc.secure_channel(
                server_url,
                credentials=grpc.ssl_channel_credentials(
                    **{
                        k: load_from_file(v) if isinstance(v, str) else v
                        for k, v in ssl_client_credentials.items()
                    }
                ),
                options=channel_options,
                compression=compression,
            )
        return grpc.insecure_channel(
            server_url, options=channel_options, compression=compression
        )

    @staticmethod
    def wait_until_server_ready(
        host: str,
        port: int,
        timeout: float = 30,
        check_interval: int = 1,
        # set kwargs here to omit gRPC kwargs
        **kwargs: t.Any,
    ) -> None:
        protocol_version = kwargs.get("protocol_version", LATEST_PROTOCOL_VERSION)

        with GrpcClient._create_channel(
            f"{host}:{port}",
            ssl=kwargs.get("ssl", False),
            ssl_client_credentials=kwargs.get("ssl_client_credentials", None),
            channel_options=kwargs.get("channel_options", None),
            compression=kwargs.get("compression", None),
        ) as channel:
            rpc = channel.unary_unary(
                "/grpc.health.v1.Health/Check",
                request_serializer=pb_health.HealthCheckRequest.SerializeToString,
                response_deserializer=pb_health.HealthCheckResponse.FromString,
            )

            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = t.cast(
                        pb_health.HealthCheckResponse,
                        rpc(
                            pb_health.HealthCheckRequest(
                                service=f"bentoml.grpc.{protocol_version}.BentoService"
                            )
                        ),
                    )
                    if response.status == pb_health.HealthCheckResponse.SERVING:
                        break
                    else:
                        time.sleep(check_interval)
                except grpc.RpcError:
                    logger.debug("Server is not ready. Retrying...")
                    time.sleep(check_interval)

            try:
                response = t.cast(
                    pb_health.HealthCheckResponse,
                    rpc(
                        pb_health.HealthCheckRequest(
                            service=f"bentoml.grpc.{protocol_version}.BentoService"
                        )
                    ),
                )
                if response.status != pb_health.HealthCheckResponse.SERVING:
                    raise TimeoutError(
                        f"Timed out waiting {timeout} seconds for server at '{host}:{port}' to be ready."
                    )
            except (grpc.RpcError, TimeoutError) as err:
                logger.error("Caught exception while connecting to %s:%s:", host, port)
                logger.error(err)
                raise

    @cached_property
    def _rpc_metadata(self) -> dict[str, dict[str, t.Any]]:
        # Currently all RPCs in BentoService are unary-unary
        # NOTE: we will set the types of the stubs to be Any.
        return {
            method: {"input_type": input_type, "output_type": output_type}
            for method, input_type, output_type in (
                (
                    self._call_rpc,
                    self._pb.Request,
                    self._pb.Response,
                ),
                (
                    f"/bentoml.grpc.{self._protocol_version}.BentoService/ServiceMetadata",
                    self._pb.ServiceMetadataRequest,
                    self._pb.ServiceMetadataResponse,
                ),
                (
                    "/grpc.health.v1.Health/Check",
                    pb_health.HealthCheckRequest,
                    pb_health.HealthCheckResponse,
                ),
            )
        }

    @cached_property
    def _rpc_methods(self) -> dict[str, t.Callable[..., "Response"]]:
        def make_sync_fn(
            method_name: str,
            input_type: t.Any,
            output_type: t.Any,
        ) -> t.Callable[..., "Response"]:
            rpc = self.channel.unary_unary(
                method_name,
                request_serializer=input_type.SerializeToString,
                response_deserializer=output_type.FromString,
            )

            def fn(
                channel_kwargs: t.Dict[str, t.Any],
                method_kwargs: t.Dict[str, t.Any],
            ) -> Response:
                return t.cast(
                    "Response",
                    rpc(input_type(**method_kwargs), **channel_kwargs),
                )

            return fn

        return {
            method_name: make_sync_fn(
                method_name,
                input_type=metadata["input_type"],
                output_type=metadata["output_type"],
            )
            for method_name, metadata in self._rpc_metadata.items()
        }

    def health(self, service_name: str, *, timeout: int = 30) -> t.Any:
        return self._rpc_methods["/grpc.health.v1.Health/Check"](
            method_kwargs={"service": service_name},
            channel_kwargs={"timeout": timeout},
        )

    @staticmethod
    def _split_channel_args(
        **kwargs: t.Any,
    ) -> tuple[t.Dict[str, t.Any], t.Dict[str, t.Any]]:
        channel_kwarg_names = (
            "timeout",
            "metadata",
            "credentials",
            "wait_for_ready",
            "compression",
        )
        channel_kwargs: t.Dict[str, t.Any] = {}
        other_kwargs: t.Dict[str, t.Any] = {}
        for k, v in kwargs.items():
            if k in channel_kwarg_names:
                channel_kwargs[k] = v
            else:
                other_kwargs[k] = v
        return other_kwargs, channel_kwargs

    def _call(
        self,
        inp: t.Any = None,
        *,
        _bentoml_api: InferenceAPI[t.Any],
        **attrs: t.Any,
    ):
        if _bentoml_api.multi_input:
            if inp is not None:
                raise BentoMLException(
                    f"'{_bentoml_api.name}' takes multiple inputs; all inputs must be passed as keyword arguments."
                )
            serialized_req = asyncio.run(_bentoml_api.input.to_proto(attrs))
        else:
            serialized_req = asyncio.run(_bentoml_api.input.to_proto(inp))

        # A call includes api_name and given proto_fields
        api_fn = {v: k for k, v in self._svc.apis.items()}
        kwargs, channel_kwargs = self._split_channel_args(**attrs)
        kwargs.update(
            {
                "api_name": api_fn[_bentoml_api],
                _bentoml_api.input.proto_fields[0]: serialized_req,
            },
        )

        if self._call_rpc not in self._rpc_methods:
            raise ValueError(
                f"'{self._call_rpc}' is a yet supported rpc. Current supported are: {self._rpc_metadata}"
            )
        proto = self._rpc_methods[self._call_rpc](
            channel_kwargs=channel_kwargs,
            method_kwargs=kwargs,
        )

        return asyncio.run(
            _bentoml_api.output.from_proto(getattr(proto, proto.WhichOneof("content")))
        )

    @classmethod
    def from_url(cls, server_url: str, **kwargs: t.Any) -> SyncGrpcClient:
        protocol_version = kwargs.get("protocol_version", LATEST_PROTOCOL_VERSION)

        # Since v1, we introduce a ServiceMetadata rpc to retrieve bentoml.Service metadata.
        # then `client.predict` or `client.classify` won't be available.
        # client.Call will still persist for both protocol version.
        if parse(protocol_version) < parse("v1"):
            exception_message = [
                f"Using protocol version {protocol_version} older than v1. 'bentoml.client.Client' will only support protocol version v1 onwards. To create client with protocol version '{protocol_version}', do the following:\n"
                f"""\

from bentoml.grpc.utils import import_generated_stubs, import_grpc

pb, services = import_generated_stubs("{protocol_version}")

grpc, _ = import_grpc()

def run():
    with grpc.insecure_channel("localhost:3000") as channel:
        stubs = services.BentoServiceStub(channel)
        req = stubs.Call(
            request=pb.Request(
                api_name="predict",
                ndarray=pb.NDArray(
                    dtype=pb.NDArray.DTYPE_FLOAT,
                    shape=(1, 4),
                    float_values=[5.9, 3, 5.1, 1.8],
                    ),
                )
            )
            print(req)

if __name__ == '__main__':
    run()
"""
            ]
            raise BentoMLException("\n".join(exception_message))
        pb, _ = import_generated_stubs(protocol_version)

        with GrpcClient._create_channel(
            server_url.replace(r"localhost", "0.0.0.0"),
            ssl=kwargs.get("ssl", False),
            ssl_client_credentials=kwargs.get("ssl_client_credentials", None),
            channel_options=kwargs.get("channel_options", None),
            compression=kwargs.get("compression", None),
        ) as channel:
            # create an insecure channel to invoke ServiceMetadata rpc
            metadata = t.cast(
                "ServiceMetadataResponse",
                channel.unary_unary(
                    f"/bentoml.grpc.{protocol_version}.BentoService/ServiceMetadata",
                    request_serializer=pb.ServiceMetadataRequest.SerializeToString,
                    response_deserializer=pb.ServiceMetadataResponse.FromString,
                )(pb.ServiceMetadataRequest()),
            )
        dummy_service = Service(metadata.name)

        for api in metadata.apis:
            try:
                dummy_service.apis[api.name] = InferenceAPI[t.Any](
                    None,
                    io_descriptors.from_spec(
                        {
                            "id": api.input.descriptor_id,
                            "args": _json_format.MessageToDict(
                                api.input.attributes
                            ).get("args", None),
                        }
                    ),
                    io_descriptors.from_spec(
                        {
                            "id": api.output.descriptor_id,
                            "args": _json_format.MessageToDict(
                                api.output.attributes
                            ).get("args", None),
                        }
                    ),
                    name=api.name,
                    doc=api.docs,
                )
            except BentoMLException as e:
                logger.error("Failed to instantiate client for API %s: ", api.name, e)

        return cls(server_url, dummy_service, **kwargs)

    def close(self):
        self.channel.close()
        return super().close()
