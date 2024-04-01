from __future__ import annotations

import functools
import logging
import typing as t
from abc import ABC
from abc import abstractmethod
from http.client import BadStatusLine

from ...exceptions import BentoMLException
from ..service.inference_api import InferenceAPI

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from types import TracebackType

    from ..service import Service
    from .grpc import AsyncGrpcClient
    from .grpc import GrpcClient
    from .grpc import SyncGrpcClient
    from .http import AsyncHTTPClient
    from .http import HTTPClient
    from .http import SyncHTTPClient


class Client(ABC):
    server_url: str
    _svc: Service
    endpoints: list[str]

    _sync_client: SyncClient
    _async_client: AsyncClient

    def __init__(self, svc: Service, server_url: str):
        logger.warning(
            "Client is deprecated and will be removed in BentoML 2.0, please use AsyncClient or SyncClient instead."
        )
        self._svc = svc
        self.server_url = server_url

        if len(svc.apis) == 0:
            raise BentoMLException("No APIs were found when constructing client.")

        self.endpoints = []
        for name in self._svc.apis.keys():
            self.endpoints.append(name)

            if not hasattr(self, name):
                setattr(self, name, functools.partial(self.call, bentoml_api_name=name))

            if not hasattr(self, f"async_{name}"):
                setattr(
                    self,
                    f"async_{name}",
                    functools.partial(self.async_call, bentoml_api_name=name),
                )

    def call(
        self,
        bentoml_api_name: str,
        inp: t.Any = None,
        **kwargs: t.Any,
    ) -> t.Any:
        return self._sync_client.call(
            inp=inp, bentoml_api_name=bentoml_api_name, **kwargs
        )

    async def async_call(
        self, bentoml_api_name: str, inp: t.Any = None, **kwargs: t.Any
    ) -> t.Any:
        return await self._async_client.call(
            inp=inp, bentoml_api_name=bentoml_api_name, **kwargs
        )

    @staticmethod
    def wait_until_server_ready(
        host: str, port: int, timeout: float = 30, **kwargs: t.Any
    ) -> None:
        SyncClient.wait_until_server_ready(host, port, timeout, **kwargs)

    @staticmethod
    async def async_wait_until_server_ready(
        host: str, port: int, timeout: float = 30, **kwargs: t.Any
    ) -> None:
        await AsyncClient.wait_until_server_ready(host, port, timeout, **kwargs)

    @t.overload
    @staticmethod
    def from_url(
        server_url: str, *, kind: None | t.Literal["auto"] = ...
    ) -> Client: ...

    @t.overload
    @staticmethod
    def from_url(server_url: str, *, kind: t.Literal["http"] = ...) -> HTTPClient: ...

    @t.overload
    @staticmethod
    def from_url(server_url: str, *, kind: t.Literal["grpc"] = ...) -> GrpcClient: ...

    @staticmethod
    def from_url(
        server_url: str,
        *,
        kind: t.Literal["auto", "http", "grpc"] | None = None,
        **kwargs: t.Any,
    ) -> Client:
        return SyncClient.from_url(server_url, kind=kind, **kwargs)

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return self._sync_client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return await self._async_client.close()


class AsyncClient(ABC):
    server_url: str
    _svc: Service
    endpoints: list[str]

    def __init__(self, svc: Service, server_url: str):
        self._svc = svc
        self.server_url = server_url

        if len(svc.apis) == 0:
            raise BentoMLException("No APIs were found when constructing client.")

        self.endpoints = []
        for name, api in self._svc.apis.items():
            self.endpoints.append(name)

            if not hasattr(self, name):
                setattr(
                    self,
                    name,
                    functools.partial(self._call, _bentoml_api=api),
                )

    async def call(
        self, bentoml_api_name: str, inp: t.Any = None, **kwargs: t.Any
    ) -> t.Any:
        return await self._call(
            inp, _bentoml_api=self._svc.apis[bentoml_api_name], **kwargs
        )

    @abstractmethod
    async def _call(
        self, inp: t.Any = None, *, _bentoml_api: InferenceAPI[t.Any], **kwargs: t.Any
    ) -> t.Any:
        raise NotImplementedError()

    async def close(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        await self.close()

    @staticmethod
    async def wait_until_server_ready(
        host: str, port: int, timeout: float = 30, **kwargs: t.Any
    ) -> None:
        try:
            from .http import AsyncHTTPClient

            await AsyncHTTPClient.wait_until_server_ready(host, port, timeout, **kwargs)
        except BadStatusLine:
            # when address is a RPC
            from .grpc import AsyncGrpcClient

            await AsyncGrpcClient.wait_until_server_ready(host, port, timeout, **kwargs)
        except Exception as err:
            # caught all other exceptions
            logger.error("Failed to connect to server %s:%s", host, port)
            logger.error(err)
            raise

    @t.overload
    @classmethod
    async def from_url(
        cls, server_url: str, *, kind: None | t.Literal["auto"] = ...
    ) -> AsyncGrpcClient | AsyncHTTPClient: ...

    @t.overload
    @classmethod
    async def from_url(
        cls, server_url: str, *, kind: t.Literal["http"] = ...
    ) -> AsyncHTTPClient: ...

    @t.overload
    @classmethod
    async def from_url(
        cls, server_url: str, *, kind: t.Literal["grpc"] = ...
    ) -> AsyncGrpcClient: ...

    @classmethod
    async def from_url(
        cls,
        server_url: str,
        *,
        kind: t.Literal["auto", "http", "grpc"] | None = None,
        **kwargs: t.Any,
    ) -> AsyncClient:
        if kind is None or kind == "auto":
            try:
                from .http import AsyncHTTPClient

                return await AsyncHTTPClient.from_url(server_url, **kwargs)
            except BadStatusLine:
                from .grpc import AsyncGrpcClient

                return await AsyncGrpcClient.from_url(server_url, **kwargs)
            except Exception as e:  # pylint: disable=broad-except
                raise BentoMLException(
                    f"Failed to create a BentoML client from given URL '{server_url}': {e} ({e.__class__.__name__})"
                ) from e
        elif kind == "http":
            from .http import AsyncHTTPClient

            return await AsyncHTTPClient.from_url(server_url, **kwargs)
        elif kind == "grpc":
            from .grpc import AsyncGrpcClient

            return await AsyncGrpcClient.from_url(server_url, **kwargs)
        else:
            raise BentoMLException(
                f"Invalid client kind '{kind}'. Must be one of 'http', 'grpc', or 'auto'."
            )


class SyncClient(Client):
    server_url: str
    _svc: Service
    endpoints: list[str]

    def __init__(self, svc: Service, server_url: str):
        self._svc = svc
        self.server_url = server_url

        if len(svc.apis) == 0:
            raise BentoMLException("No APIs were found when constructing client.")

        self.endpoints = []
        for name, api in self._svc.apis.items():
            self.endpoints.append(name)

            if not hasattr(self, name):
                setattr(
                    self,
                    name,
                    functools.partial(self._call, _bentoml_api=api),
                )

    def call(self, bentoml_api_name: str, inp: t.Any = None, **kwargs: t.Any) -> t.Any:
        return self._call(inp, _bentoml_api=self._svc.apis[bentoml_api_name], **kwargs)

    @abstractmethod
    def _call(
        self, inp: t.Any = None, *, _bentoml_api: InferenceAPI[t.Any], **kwargs: t.Any
    ) -> t.Any:
        raise NotImplementedError()

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.close()

    @staticmethod
    def wait_until_server_ready(
        host: str, port: int, timeout: float = 30, **kwargs: t.Any
    ) -> None:
        try:
            from .http import SyncHTTPClient

            SyncHTTPClient.wait_until_server_ready(host, port, timeout, **kwargs)
        except BadStatusLine:
            # when address is a RPC
            from .grpc import SyncGrpcClient

            SyncGrpcClient.wait_until_server_ready(host, port, timeout, **kwargs)
        except Exception as err:
            # caught all other exceptions
            logger.error("Failed to connect to server %s:%s", host, port)
            logger.error(err)
            raise

    @t.overload
    @classmethod
    def from_url(
        cls, server_url: str, *, kind: None | t.Literal["auto"] = ...
    ) -> SyncGrpcClient | SyncHTTPClient: ...

    @t.overload
    @classmethod
    def from_url(
        cls, server_url: str, *, kind: t.Literal["http"] = ...
    ) -> SyncHTTPClient: ...

    @t.overload
    @classmethod
    def from_url(
        cls, server_url: str, *, kind: t.Literal["grpc"] = ...
    ) -> SyncGrpcClient: ...

    @classmethod
    def from_url(
        cls,
        server_url: str,
        *,
        kind: t.Literal["auto", "http", "grpc"] | None = None,
        **kwargs: t.Any,
    ) -> SyncClient:
        if kind is None or kind == "auto":
            try:
                from .http import SyncHTTPClient

                return SyncHTTPClient.from_url(server_url, **kwargs)
            except BadStatusLine:
                from .grpc import SyncGrpcClient

                return SyncGrpcClient.from_url(server_url, **kwargs)
            except Exception as e:  # pylint: disable=broad-except
                raise BentoMLException(
                    f"Failed to create a BentoML client from given URL '{server_url}': {e} ({e.__class__.__name__})"
                ) from e
        elif kind == "http":
            from .http import SyncHTTPClient

            return SyncHTTPClient.from_url(server_url, **kwargs)
        elif kind == "grpc":
            from .grpc import SyncGrpcClient

            return SyncGrpcClient.from_url(server_url, **kwargs)
        else:
            raise BentoMLException(
                f"Invalid client kind '{kind}'. Must be one of 'http', 'grpc', or 'auto'."
            )
