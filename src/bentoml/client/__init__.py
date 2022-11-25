from __future__ import annotations

import typing as t
import asyncio
import logging
import functools
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import attr

from .. import Service
from ..exceptions import BentoMLException
from ..grpc.utils import import_grpc
from ..grpc.utils import LATEST_PROTOCOL_VERSION
from .._internal.utils import bentoml_cattr
from .._internal.utils import cached_property
from .._internal.service.inference_api import InferenceAPI

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from urllib.parse import ParseResult

    import grpc
    from grpc import aio

    from ._grpc import GrpcClient
    from ._http import HTTPClient
    from .._internal.types import PathType

    class ClientCredentials(t.TypedDict):
        root_certificates: t.NotRequired[PathType | bytes]
        private_key: t.NotRequired[PathType | bytes]
        certificate_chain: t.NotRequired[PathType | bytes]

else:
    ClientCredentials = dict

    grpc, aio = import_grpc()


@attr.define
class ClientConfig:
    http: HTTP = attr.field(
        default=attr.Factory(lambda self: self.HTTP(), takes_self=True)
    )
    grpc: GRPC = attr.field(
        default=attr.Factory(lambda self: self.GRPC(), takes_self=True)
    )

    def with_grpc_options(self, **kwargs: t.Any) -> ClientConfig:
        _self_grpc_config = kwargs.pop("_self_grpc_config", None)
        if not isinstance(_self_grpc_config, self.GRPC):
            _self_grpc_config = ClientConfig.GRPC.from_options(**kwargs)
        return attr.evolve(self, **{"grpc": _self_grpc_config})

    def with_http_options(self, **kwargs: t.Any) -> ClientConfig:
        _self_http_config = kwargs.pop("_self_http_config", None)
        if not isinstance(_self_http_config, self.HTTP):
            _self_http_config = ClientConfig.HTTP.from_options(**kwargs)
        return attr.evolve(self, **{"http": _self_http_config})

    @classmethod
    def from_options(cls, **kwargs: t.Any) -> ClientConfig:
        return bentoml_cattr.structure(kwargs, cls)

    @staticmethod
    def from_grpc_options(**kwargs: t.Any) -> GRPC:
        return ClientConfig.GRPC.from_options(**kwargs)

    @staticmethod
    def from_http_options(**kwargs: t.Any) -> HTTP:
        return ClientConfig.HTTP.from_options(**kwargs)

    def unstructure(
        self, target: t.Literal["http", "grpc", "default"] = "default"
    ) -> dict[str, t.Any]:
        if target == "default":
            targ = self
        elif target == "http":
            targ = self.http
        elif target == "grpc":
            targ = self.grpc
        else:
            raise ValueError(
                f"Invalid target: {target}. Accepted value are 'http', 'grpc', 'default'."
            )
        return bentoml_cattr.unstructure(targ)

    @attr.define
    class HTTP:
        """HTTP ClientConfig.

        .. TODO:: Add HTTP specific options here.

        """

        # forbid additional keys to prevent typos.
        __forbid_extra_keys__ = True
        # Don't omit empty field.
        __omit_if_default__ = False

        @classmethod
        def from_options(cls, **kwargs: t.Any) -> ClientConfig.HTTP:
            return bentoml_cattr.structure(kwargs, cls)

        def unstructure(self) -> dict[str, t.Any]:
            return (
                ClientConfig()
                .with_http_options(
                    _self_http_config=self,
                )
                .unstructure(target="http")
            )

    @attr.define
    class GRPC:
        """gRPC ClientConfig.

        .. code-block:: python

            from bentoml.client import ClientConfig
            from bentoml.client import Client

            config = ClientConfig.from_grpc_options(
                ssl=True,
                ssl_client_credentials={
                    "root_certificates": "path/to/cert.pem",
                    "private_key": "/path/to/key",
                },
                protocol_version="v1alpha1",
            )
            client = Client.from_url("localhost:50051", config)

        """

        # forbid additional keys to prevent typos.
        __forbid_extra_keys__ = True
        # Don't omit empty field.
        __omit_if_default__ = False

        ssl: bool = attr.field(default=False)
        channel_options: t.Optional[aio.ChannelArgumentType] = attr.field(default=None)
        compression: t.Optional[grpc.Compression] = attr.field(default=None)
        ssl_client_credentials: t.Optional[ClientCredentials] = attr.field(
            factory=lambda: ClientCredentials()
        )
        protocol_version: str = attr.field(default=LATEST_PROTOCOL_VERSION)
        interceptors: t.Optional[t.Sequence[aio.ClientInterceptor]] = attr.field(
            default=None
        )

        @classmethod
        def from_options(cls, **kwargs: t.Any) -> ClientConfig.GRPC:
            return bentoml_cattr.structure(kwargs, cls)

        def unstructure(self) -> dict[str, t.Any]:
            return (
                ClientConfig()
                .with_grpc_options(
                    _self_grpc_config=self,
                )
                .unstructure(target="grpc")
            )


if TYPE_CHECKING:
    ClientConfigT = ClientConfig | ClientConfig.HTTP | ClientConfig.GRPC


_sentinel_svc = Service("sentinel_svc")


class Client(ABC):
    server_url: str
    _svc: Service

    def __init__(self, svc: Service | None, server_url: str):
        self._svc = svc or _sentinel_svc
        self.server_url = server_url

        if svc is not None and len(svc.apis) == 0:
            raise BentoMLException("No APIs were found when constructing client.")

        # Register service method if given service is not _sentinel_svc
        # We only set _sentinel_svc if given protocol is older than v1 (for gRPC client.)
        if self._svc is not _sentinel_svc:
            for name, api in self._svc.apis.items():
                if not hasattr(self, name):
                    setattr(
                        self, name, functools.partial(self._sync_call, _bentoml_api=api)
                    )

                if not hasattr(self, f"async_{name}"):
                    setattr(
                        self,
                        f"async_{name}",
                        functools.partial(self._call, _bentoml_api=api),
                    )

    def call(self, bentoml_api_name: str, inp: t.Any = None, **kwargs: t.Any) -> t.Any:
        return self._loop.run_until_complete(
            self.async_call(bentoml_api_name, inp, **kwargs)
        )

    async def async_call(
        self, bentoml_api_name: str, inp: t.Any = None, **kwargs: t.Any
    ) -> t.Any:
        return await self._call(
            inp, _bentoml_api=self._svc.apis[bentoml_api_name], **kwargs
        )

    @t.overload
    @staticmethod
    def from_url(
        server_url: str,
        config: ClientConfigT | None = ...,
        *,
        grpc: t.Literal[False] = ...,
    ) -> HTTPClient:
        ...

    @t.overload
    @staticmethod
    def from_url(
        server_url: str,
        config: ClientConfigT | None = ...,
        *,
        grpc: t.Literal[True] = ...,
    ) -> GrpcClient:
        ...

    @staticmethod
    def from_url(
        server_url: str, config: ClientConfigT | None = None, *, grpc: bool = False
    ) -> Client:
        server_url = server_url if "://" in server_url else "http://" + server_url
        if grpc:
            from ._grpc import GrpcClient

            client_type = "grpc"
            klass = GrpcClient
        else:
            from ._http import HTTPClient

            client_type = "http"
            klass = HTTPClient

        if config is None:
            config = ClientConfig()

        # First, if config is a ClientConfig that contains both HTTP and gRPC fields, then we use
        # grpc_client boolean to determine which configset to use.
        # If config is either ClientConfig.HTTP or ClientConfig.GRPC, then we use unstructure for kwargs
        kwargs = config.unstructure()

        if isinstance(config, ClientConfig):
            # by default we will set the config to HTTP (backward compatibility)
            kwargs = config.unstructure(target=client_type)

        try:
            return getattr(klass, "_create_client")(urlparse(server_url), **kwargs)
        except Exception as e:  # pylint: disable=broad-except
            raise BentoMLException(
                f"Failed to create a BentoML client from given URL '{server_url}': {e} ({e.__class__.__name__})"
            ) from e

    @cached_property
    def _loop(self) -> asyncio.AbstractEventLoop:
        return asyncio.get_event_loop()

    def _sync_call(
        self, inp: t.Any = None, *, _bentoml_api: InferenceAPI, **kwargs: t.Any
    ):
        return self._loop.run_until_complete(
            self._call(inp, _bentoml_api=_bentoml_api, **kwargs)
        )

    @abstractmethod
    async def _call(
        self, inp: t.Any = None, *, _bentoml_api: InferenceAPI, **kwargs: t.Any
    ) -> t.Any:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _create_client(parsed: ParseResult, **kwargs: t.Any) -> Client:
        raise NotImplementedError
