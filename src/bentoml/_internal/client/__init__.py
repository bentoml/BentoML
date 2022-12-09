from __future__ import annotations

import typing as t
import asyncio
import logging
import functools
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from http.client import BadStatusLine
from urllib.parse import urlparse

from ...exceptions import BentoMLException
from ..service.inference_api import InferenceAPI

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .grpc import GrpcClient
    from .http import HTTPClient
    from ..service import Service


class Client(ABC):
    server_url: str
    _svc: Service

    def __init__(self, svc: Service, server_url: str):
        self._svc = svc
        self.server_url = server_url

        if svc is not None and len(svc.apis) == 0:
            raise BentoMLException("No APIs were found when constructing client.")

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
        return self._sync_call(
            inp, _bentoml_api=self._svc.apis[bentoml_api_name], **kwargs
        )

    async def async_call(
        self, bentoml_api_name: str, inp: t.Any = None, **kwargs: t.Any
    ) -> t.Any:
        return await self._call(
            inp, _bentoml_api=self._svc.apis[bentoml_api_name], **kwargs
        )

    @abstractmethod
    def wait_until_server_ready(
        self,
        *,
        server_url: str | None = None,
        timeout: int = 30,
        **kwargs: t.Any,
    ) -> None:
        raise NotImplementedError

    @t.overload
    @classmethod
    @abstractmethod
    def from_url(
        cls, server_url: str, *, kind: None | t.Literal["auto"] = ...
    ) -> GrpcClient | HTTPClient:
        ...

    @t.overload
    @classmethod
    @abstractmethod
    def from_url(cls, server_url: str, *, kind: t.Literal["http"] = ...) -> HTTPClient:
        ...

    @t.overload
    @classmethod
    @abstractmethod
    def from_url(cls, server_url: str, *, kind: t.Literal["grpc"] = ...) -> GrpcClient:
        ...

    @classmethod
    @abstractmethod
    def from_url(
        cls, server_url: str, *, kind: str | None = None, **kwargs: t.Any
    ) -> Client:
        url_parsed = urlparse(server_url)
        if url_parsed.scheme == "http":
            kind = "http"
        elif url_parsed.scheme == "grpc":
            kind = "grpc"

        if kind is None or kind == "auto":
            try:
                from .http import HTTPClient

                return HTTPClient.from_url(server_url, **kwargs)
            except BadStatusLine:
                from .grpc import GrpcClient

                return GrpcClient.from_url(server_url, **kwargs)
            except Exception as e:  # pylint: disable=broad-except
                raise BentoMLException(
                    f"Failed to create a BentoML client from given URL '{server_url}': {e} ({e.__class__.__name__})"
                ) from e
        elif kind == "http":
            from .http import HTTPClient

            return HTTPClient.from_url(server_url, **kwargs)
        elif kind == "grpc":
            from .grpc import GrpcClient

            return GrpcClient.from_url(server_url, **kwargs)
        else:
            raise BentoMLException(
                "Invalid client kind. Must be one of ['http', 'grpc', 'auto']"
            )

    def _sync_call(
        self, inp: t.Any = None, *, _bentoml_api: InferenceAPI, **kwargs: t.Any
    ):
        return asyncio.run(self._call(inp, _bentoml_api=_bentoml_api, **kwargs))

    @abstractmethod
    async def _call(
        self, inp: t.Any = None, *, _bentoml_api: InferenceAPI, **kwargs: t.Any
    ) -> t.Any:
        raise NotImplementedError
