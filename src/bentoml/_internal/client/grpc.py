from __future__ import annotations

import typing as t
import logging
import functools
from typing import TYPE_CHECKING

from packaging.version import parse

from . import Client
from .. import io_descriptors as io
from ..utils import LazyLoader
from ..utils import cached_property
from ..service import Service
from ...exceptions import BentoMLException
from ...grpc.utils import import_grpc
from ...grpc.utils import import_generated_stubs
from ...grpc.utils import LATEST_PROTOCOL_VERSION
from ..server.grpc_app import load_from_file
from ..service.inference_api import InferenceAPI

logger = logging.getLogger(__name__)

PROTOBUF_EXC_MESSAGE = "'protobuf' is required to use gRPC Client. Install with 'pip install bentoml[grpc]'."
REFLECTION_EXC_MESSAGE = "'grpcio-reflection' is required to use gRPC Client. Install with 'pip install bentoml[grpc-reflection]'."

if TYPE_CHECKING:

    import grpc
    from grpc import aio
    from google.protobuf import json_format as _json_format

    from ..types import PathType
    from ...grpc.v1.service_pb2 import ServiceMetadataResponse

    class ClientCredentials(t.TypedDict):
        root_certificates: t.NotRequired[PathType | bytes]
        private_key: t.NotRequired[PathType | bytes]
        certificate_chain: t.NotRequired[PathType | bytes]

else:
    ClientCredentials = dict
    _json_format = LazyLoader(
        "_json_format",
        globals(),
        "google.protobuf.json_format",
        exc_msg=PROTOBUF_EXC_MESSAGE,
    )
    grpc, aio = import_grpc()

_INDENTATION = " " * 4

# TODO: xDS support
class GrpcClient(Client):
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
    ):
        self._pb, self._services = import_generated_stubs(protocol_version)

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
        super().__init__(svc, server_url)

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

    @cached_property
    def _rpc_handler_mapping(self):
        # Currently all RPCs in BentoService are unary-unary
        return {
            method: {
                "handler": self.channel.unary_unary(
                    method=method,
                    request_serializer=input_type.SerializeToString,
                    response_deserializer=output_type.FromString,
                ),
                "input_type": input_type,
                "output_type": output_type,
            }
            for method, input_type, output_type in (
                (
                    f"/bentoml.grpc.{self._protocol_version}.BentoService/Call",
                    self._pb.Request,
                    self._pb.Response,
                ),
                (
                    f"/bentoml.grpc.{self._protocol_version}.BentoService/ServiceMetadata",
                    self._pb.ServiceMetadataRequest,
                    self._pb.ServiceMetadataResponse,
                ),
            )
        }

    async def _invoke(self, method_name: str, **attrs: t.Any):
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
        if method_name not in self._rpc_handler_mapping:
            raise ValueError(
                f"'{method_name}' is a yet supported rpc. Current supported are: {list(self._rpc_handler_mapping.keys())}"
            )
        rpc_handler = self._rpc_handler_mapping[method_name]

        return await t.cast(
            t.Awaitable[t.Any],
            rpc_handler["handler"](
                rpc_handler["input_type"](**attrs), **channel_kwargs
            ),
        )

    async def _call(
        self,
        inp: t.Any = None,
        *,
        _bentoml_api: InferenceAPI,
        **attrs: t.Any,
    ) -> t.Any:
        if self.channel.get_state() != grpc.ChannelConnectivity.READY:
            # create a blocking call to wait til channel is ready.
            await self.channel.channel_ready()

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
        _rev_apis = {v: k for k, v in self._svc.apis.items()}
        return await fn(
            f"/bentoml.grpc.{self._protocol_version}.BentoService/Call",
            **{
                "api_name": _rev_apis[_bentoml_api],
                _bentoml_api.input._proto_fields[0]: serialized_req,
            },
        )

    @classmethod
    def from_url(cls, server_url: str, **kwargs: t.Any) -> GrpcClient:
        protocol_version = kwargs.get("protocol_version", LATEST_PROTOCOL_VERSION)
        ssl = kwargs.get("ssl", False)
        ssl_client_credentials = kwargs.get("ssl_client_credentials", None)

        # Since v1, we introduce a ServiceMetadata rpc to retrieve bentoml.Service metadata.
        # then `client.predict` or `client.classify` won't be available.
        # client.Call will still persist for both protocol version.
        if parse(protocol_version) < parse("v1"):
            exception_message = [
                f"Using protocol version {protocol_version} older than v1. 'bentoml.client.Client' will only support protocol version v1 onwards. To create client with protocol version '{protocol_version}', do the following:\n"
                """\

from bentoml.grpc.utils import import_generated_stubs, import_grpc

pb, services = import_generated_stubs("v1alpha1")

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

        if ssl:
            assert (
                ssl_client_credentials is not None
            ), "'ssl=True' requires 'credentials'"
            channel = grpc.secure_channel(
                server_url,
                credentials=grpc.ssl_channel_credentials(
                    **{
                        k: load_from_file(v) if isinstance(v, str) else v
                        for k, v in ssl_client_credentials.items()
                    }
                ),
                options=kwargs.get("channel_options", None),
                compression=kwargs.get("compression", None),
            )
        else:
            channel = grpc.insecure_channel(
                server_url,
                options=kwargs.get("channel_options", None),
                compression=kwargs.get("compression", None),
            )

        # create an insecure channel to invoke ServiceMetadata rpc
        with channel:
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
                dummy_service.apis[api.name] = InferenceAPI(
                    None,
                    io.from_spec(
                        {
                            "id": api.input.descriptor_id,
                            "args": _json_format.MessageToDict(
                                api.input.attributes
                            ).get("args", None),
                        }
                    ),
                    io.from_spec(
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
