from __future__ import annotations

import time
import typing as t
import asyncio
import logging

from . import Client
from .. import io_descriptors
from ..utils import LazyLoader
from ..service import Service
from ...exceptions import BentoMLException
from ...grpc.utils import import_grpc
from ...grpc.utils import import_generated_stubs
from ...grpc.utils import LATEST_PROTOCOL_VERSION
from ..server.grpc_app import load_from_file
from ..service.inference_api import InferenceAPI

logger = logging.getLogger(__name__)

PROTOBUF_EXC_MESSAGE = "'protobuf' is required to use gRPC Client. Install with 'pip install bentoml[grpc]'."

if t.TYPE_CHECKING:
    import grpc
    from grpc import aio
    from grpc_health.v1 import health_pb2 as pb_health
    from grpc_health.v1 import health_pb2_grpc as services_health
    from google.protobuf import json_format as _json_format

    from ..types import PathType

    class ClientCredentials(t.TypedDict):
        root_certificates: t.NotRequired[PathType | bytes]
        private_key: t.NotRequired[PathType | bytes]
        certificate_chain: t.NotRequired[PathType | bytes]

else:
    grpc, aio = import_grpc()
    pb_health = LazyLoader("pb_health", globals(), "grpc_health.v1.health_pb2")
    services_health = LazyLoader(
        "services_health", globals(), "grpc_health.v1.health_pb2_grpc"
    )
    _json_format = LazyLoader(
        "_json_format",
        globals(),
        "google.protobuf.json_format",
        exc_msg=PROTOBUF_EXC_MESSAGE,
    )


def _create_async_channel(
    server_url: str,
    ssl: bool = False,
    ssl_client_credentials: ClientCredentials | None = None,
    options: t.Any | None = None,
    compression: grpc.Compression | None = None,
    interceptors: t.Sequence[aio.ClientInterceptor] | None = None,
) -> aio._channel.Channel:
    if ssl:
        assert (
            ssl_client_credentials is not None
        ), "'ssl=True' requires 'ssl_client_credentials'"
        return aio.secure_channel(
            server_url,
            credentials=grpc.ssl_channel_credentials(
                **{
                    k: load_from_file(v) if isinstance(v, str) else v
                    for k, v in ssl_client_credentials.items()
                }
            ),
            options=options,
            compression=compression,
            interceptors=interceptors,
        )
    else:
        return aio.insecure_channel(
            server_url,
            options=options,
            compression=compression,
            interceptors=interceptors,
        )


def _ensure_exec_coro(coro: t.Coroutine[t.Any, t.Any, t.Any]) -> t.Any:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()
    else:
        return loop.run_until_complete(coro)


# TODO: xDS support
class GrpcClient(Client):
    def __init__(
        self,
        server_url: str,
        svc: Service,
        # gRPC specific options
        ssl: bool = False,
        options: aio.ChannelArgumentType | None = None,
        interceptors: t.Sequence[aio.ClientInterceptor] | None = None,
        compression: grpc.Compression | None = None,
        ssl_client_credentials: ClientCredentials | None = None,
        *,
        protocol_version: str = LATEST_PROTOCOL_VERSION,
        **kwargs: t.Any,
    ):
        self._pb, self._services = import_generated_stubs(protocol_version)

        self._compression = compression
        self._options = options
        self._interceptors = interceptors
        self._ssl = ssl
        self._ssl_client_credentials = ssl_client_credentials

        super().__init__(svc, server_url)

    def create_channel(self):
        return _create_async_channel(
            self.server_url,
            ssl=self._ssl,
            ssl_client_credentials=self._ssl_client_credentials,
            options=self._options,
            compression=self._compression,
            interceptors=self._interceptors,
        )

    @staticmethod
    def wait_until_server_ready(
        host: str,
        port: int,
        timeout: int = 30,
        check_interval: int = 1,
        # set kwargs here to omit gRPC kwargs
        **kwargs: t.Any,
    ) -> None:
        with grpc.insecure_channel(
            f"{host.replace(r'localhost', '0.0.0.0')}:{port}",
            options=kwargs.get("options", None),
            compression=kwargs.get("compression", None),
        ) as channel:
            req = pb_health.HealthCheckRequest()
            req.service = ""
            health_stub = services_health.HealthStub(channel)
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    resp = health_stub.Check(req)
                    if resp.status == pb_health.HealthCheckResponse.SERVING:
                        break
                    else:
                        time.sleep(check_interval)
                except grpc.RpcError:
                    logger.debug("Waiting for server to be ready...")
                    time.sleep(check_interval)
            try:
                resp = health_stub.Check(req)
                if resp.status != pb_health.HealthCheckResponse.SERVING:
                    raise TimeoutError(
                        f"Timed out waiting {timeout} seconds for server at '{host}:{port}' to be ready."
                    )
            except grpc.RpcError as err:
                logger.error("Caught RpcError while connecting to %s:%s:\n", host, port)
                logger.error(err)
                raise

    def sync_health(self, service_name: str, *, timeout: int = 30) -> t.Any:
        return _ensure_exec_coro(self.health(service_name, timeout=timeout))

    async def health(self, service_name: str, *, timeout: int = 30) -> t.Any:
        async with self.create_channel() as channel:
            req = pb_health.HealthCheckRequest()
            req.service = service_name
            health_stub = services_health.HealthStub(channel)
            return await health_stub.Check(req, timeout=timeout)

    def _sync_call(
        self, inp: t.Any = None, *, _bentoml_api: InferenceAPI, **attrs: t.Any
    ) -> t.Any:
        return _ensure_exec_coro(self._call(inp, _bentoml_api=_bentoml_api, **attrs))

    async def _call(
        self, inp: t.Any = None, *, _bentoml_api: InferenceAPI, **attrs: t.Any
    ) -> t.Any:
        channel_kwargs = {
            k: attrs.pop(f"_grpc_channel_{k}", None)
            for k in {
                "timeout",
                "metadata",
                "credentials",
                "wait_for_ready",
                "compression",
            }
        }

        channel = self.create_channel()
        state = channel.get_state(try_to_connect=True)
        if state != grpc.ChannelConnectivity.READY:
            # create a blocking call to wait til channel is ready.
            await channel.channel_ready()

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

        async with channel:
            stubs = self._services.BentoServiceStub(channel)
            req = self._pb.Request(
                **{
                    "api_name": api_fn[_bentoml_api],
                    _bentoml_api.input.proto_fields[0]: serialized_req,
                }
            )
            return await stubs.Call(req, **channel_kwargs)

    @classmethod
    def from_url(cls, server_url: str, **kwargs: t.Any) -> GrpcClient:
        protocol_version = kwargs.get("protocol_version", LATEST_PROTOCOL_VERSION)
        pb, services = import_generated_stubs(protocol_version)

        with grpc.insecure_channel(
            server_url.replace(r"localhost", "0.0.0.0"),
            options=kwargs.get("options", None),
        ) as channel:
            stubs = services.BentoServiceStub(channel)
            # create an insecure channel to invoke ServiceMetadata rpc
            metadata = stubs.ServiceMetadata(pb.ServiceMetadataRequest())

        dummy_service = Service(metadata.name)

        for api in metadata.apis:
            try:
                dummy_service.apis[api.name] = InferenceAPI(
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
                raise

        return cls(server_url, dummy_service, **kwargs)
