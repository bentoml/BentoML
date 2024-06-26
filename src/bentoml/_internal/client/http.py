from __future__ import annotations

import asyncio
import json
import logging
import time
import typing as t
from functools import cached_property

import httpx
import starlette.datastructures
import starlette.requests

from ...exceptions import BentoMLException
from ...exceptions import RemoteException
from .. import io_descriptors as io
from ..service import Service
from ..service.inference_api import InferenceAPI
from . import AsyncClient
from . import Client
from . import SyncClient

logger = logging.getLogger(__name__)


class HTTPClient(Client):
    def __init__(self, svc: Service, server_url: str):
        self._sync_client = SyncHTTPClient(svc=svc, server_url=server_url)
        self._async_client = AsyncHTTPClient(svc=svc, server_url=server_url)
        super().__init__(svc, server_url)


class AsyncHTTPClient(AsyncClient):
    @cached_property
    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.server_url, timeout=300)

    @staticmethod
    async def wait_until_server_ready(
        host: str,
        port: int,
        timeout: float = 30,
        check_interval: int = 1,
        # set kwargs here to omit gRPC kwargs
        **kwargs: t.Any,
    ) -> None:
        host = host if "://" in host else "http://" + host
        start_time = time.time()

        logger.debug("Waiting for host %s to be ready.", f"{host}:{port}")
        while time.time() - start_time < timeout:
            try:
                async with httpx.AsyncClient(base_url=f"{host}:{port}") as session:
                    resp = await session.get("/readyz")
                    if resp.status_code == 200:
                        break
                    else:
                        await asyncio.sleep(check_interval)
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.HTTPStatusError,
            ):
                logger.debug("Server is not ready. Retrying...")
                await asyncio.sleep(check_interval)

        # try to connect one more time and raise exception.
        try:
            async with httpx.AsyncClient(base_url=f"{host}:{port}") as session:
                resp = await session.get("/readyz")
                if resp.status_code != 200:
                    raise TimeoutError(
                        f"Timed out waiting {timeout} seconds for server at '{host}:{port}' to be ready."
                    )
        except (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.HTTPStatusError,
        ) as err:
            logger.error("Timed out while connecting to %s:%s:", host, port)
            logger.error(err)
            raise

    async def health(self) -> httpx.Response:
        return await self.client.get("/readyz")

    @classmethod
    async def from_url(cls, server_url: str, **kwargs: t.Any) -> AsyncHTTPClient:
        server_url = server_url if "://" in server_url else "http://" + server_url

        async with httpx.AsyncClient(base_url=server_url) as session:
            resp = await session.get("/docs.json")
            if resp.status_code != 200:
                raise RemoteException(
                    f"Failed to get OpenAPI schema from the server: {resp.status_code} {resp.reason_phrase}:\n{await resp.aread()}"
                )
            openapi_spec = json.loads(await resp.aread())

        dummy_service = Service(openapi_spec["info"]["title"])

        for route, spec in openapi_spec["paths"].items():
            for meth_spec in spec.values():
                if "tags" in meth_spec and "Service APIs" in meth_spec["tags"]:
                    if "x-bentoml-io-descriptor" not in meth_spec["requestBody"]:
                        # TODO: better message stating min version for from_url to work
                        raise BentoMLException(
                            f"Malformed BentoML spec received from BentoML server {server_url}"
                        )
                    if "x-bentoml-io-descriptor" not in meth_spec["responses"]["200"]:
                        raise BentoMLException(
                            f"Malformed BentoML spec received from BentoML server {server_url}"
                        )
                    if "x-bentoml-name" not in meth_spec:
                        raise BentoMLException(
                            f"Malformed BentoML spec received from BentoML server {server_url}"
                        )
                    try:
                        api = InferenceAPI[t.Any](
                            None,
                            io.from_spec(
                                meth_spec["requestBody"]["x-bentoml-io-descriptor"]
                            ),
                            io.from_spec(
                                meth_spec["responses"]["200"]["x-bentoml-io-descriptor"]
                            ),
                            name=meth_spec["x-bentoml-name"],
                            doc=meth_spec["description"],
                            route=route.lstrip("/"),
                        )
                        dummy_service.apis[meth_spec["x-bentoml-name"]] = api
                    except BentoMLException as e:
                        logger.error(
                            "Failed to instantiate client for API %s: %s",
                            meth_spec["x-bentoml-name"],
                            e,
                        )

        return cls(dummy_service, server_url)

    async def _call(
        self, inp: t.Any = None, *, _bentoml_api: InferenceAPI[t.Any], **kwargs: t.Any
    ) -> t.Any:
        # All gRPC kwargs should be poped out.
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_grpc_")}
        api = _bentoml_api

        if api.multi_input:
            if inp is not None:
                raise BentoMLException(
                    f"'{api.name}' takes multiple inputs; all inputs must be passed as keyword arguments."
                )
            fake_resp = await api.input.to_http_response(kwargs, None)
        else:
            fake_resp = await api.input.to_http_response(inp, None)

        # TODO: Temporary workaround before moving everything to StreamingResponse
        if isinstance(fake_resp, starlette.responses.StreamingResponse):
            req_body = "".join([s async for s in fake_resp.body_iterator])
        else:
            req_body = fake_resp.body

        resp = await self.client.post(
            api.route,
            content=req_body,
            headers={"content-type": fake_resp.headers["content-type"]},
        )
        if resp.status_code != 200:
            raise BentoMLException(
                f"Error making request: {resp.status_code}: {str(await resp.aread())}"
            )

        fake_req = starlette.requests.Request(scope={"type": "http"})
        headers = starlette.datastructures.Headers(headers=resp.headers)
        fake_req._body = await resp.aread()
        # Request.headers sets a _headers variable. We will need to set this
        # value to our fake request object.
        fake_req._headers = headers  # type: ignore (request._headers is property)

        return await api.output.from_http_request(fake_req)

    async def close(self):
        await self.client.aclose()
        return await super().close()


class SyncHTTPClient(SyncClient):
    @cached_property
    def client(self) -> httpx.Client:
        return httpx.Client(base_url=self.server_url, timeout=300)

    @staticmethod
    def wait_until_server_ready(
        host: str,
        port: int,
        timeout: float = 30,
        check_interval: int = 1,
        # set kwargs here to omit gRPC kwargs
        **kwargs: t.Any,
    ) -> None:
        host = host if "://" in host else "http://" + host
        start_time = time.time()

        logger.debug("Waiting for host %s to be ready.", f"{host}:{port}")
        while time.time() - start_time < timeout:
            try:
                status = httpx.get(f"{host}:{port}/readyz").status_code
                if status == 200:
                    break
                else:
                    time.sleep(check_interval)
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.HTTPStatusError,
            ):
                logger.debug("Server is not ready. Retrying...")

        # try to connect one more time and raise exception.
        try:
            status = httpx.get(f"{host}:{port}/readyz").status_code
            if status != 200:
                raise TimeoutError(
                    f"Timed out waiting {timeout} seconds for server at '{host}:{port}' to be ready."
                )
        except (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.HTTPStatusError,
        ) as err:
            logger.error("Timed out while connecting to %s:%s:", host, port)
            logger.error(err)
            raise

    def health(self) -> httpx.Response:
        return self.client.get("/readyz")

    @classmethod
    def from_url(cls, server_url: str, **kwargs: t.Any) -> SyncHTTPClient:
        server_url = server_url if "://" in server_url else "http://" + server_url

        with httpx.Client(base_url=server_url) as session:
            resp = session.get("docs.json")
            if resp.status_code != 200:
                raise RemoteException(
                    f"Failed to get OpenAPI schema from the server: {resp.status_code} {resp.reason_phrase}:\n{resp.content}"
                )
            openapi_spec = json.loads(resp.content)

        dummy_service = Service(openapi_spec["info"]["title"])

        for route, spec in openapi_spec["paths"].items():
            for meth_spec in spec.values():
                if "tags" in meth_spec and "Service APIs" in meth_spec["tags"]:
                    if "x-bentoml-io-descriptor" not in meth_spec["requestBody"]:
                        # TODO: better message stating min version for from_url to work
                        raise BentoMLException(
                            f"Malformed BentoML spec received from BentoML server {server_url}"
                        )
                    if "x-bentoml-io-descriptor" not in meth_spec["responses"]["200"]:
                        raise BentoMLException(
                            f"Malformed BentoML spec received from BentoML server {server_url}"
                        )
                    if "x-bentoml-name" not in meth_spec:
                        raise BentoMLException(
                            f"Malformed BentoML spec received from BentoML server {server_url}"
                        )
                    try:
                        api = InferenceAPI[t.Any](
                            None,
                            io.from_spec(
                                meth_spec["requestBody"]["x-bentoml-io-descriptor"]
                            ),
                            io.from_spec(
                                meth_spec["responses"]["200"]["x-bentoml-io-descriptor"]
                            ),
                            name=meth_spec["x-bentoml-name"],
                            doc=meth_spec["description"],
                            route=route.lstrip("/"),
                        )
                        dummy_service.apis[meth_spec["x-bentoml-name"]] = api
                    except BentoMLException as e:
                        logger.error(
                            "Failed to instantiate client for API %s: %s",
                            meth_spec["x-bentoml-name"],
                            e,
                        )

        return cls(dummy_service, server_url)

    def _call(
        self, inp: t.Any = None, *, _bentoml_api: InferenceAPI[t.Any], **kwargs: t.Any
    ) -> t.Any:
        # All gRPC kwargs should be poped out.
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_grpc_")}
        api = _bentoml_api

        if api.multi_input:
            if inp is not None:
                raise BentoMLException(
                    f"'{api.name}' takes multiple inputs; all inputs must be passed as keyword arguments."
                )
            # TODO: remove asyncio run after descriptor rework
            fake_resp = asyncio.run(api.input.to_http_response(kwargs, None))
        else:
            fake_resp = asyncio.run(api.input.to_http_response(inp, None))

        # TODO: Temporary workaround before moving everything to StreamingResponse
        if isinstance(fake_resp, starlette.responses.StreamingResponse):

            async def get_body():
                return "".join([s async for s in fake_resp.body_iterator])

            req_body = asyncio.run(get_body())
        else:
            req_body = fake_resp.body

        resp = self.client.post(
            api.route,
            content=req_body,
            headers={"content-type": fake_resp.headers["content-type"]},
        )
        if resp.status_code != 200:
            raise BentoMLException(
                f"Error making request: {resp.status_code}: {str(resp.content)}"
            )

        fake_req = starlette.requests.Request(scope={"type": "http"})
        headers = starlette.datastructures.Headers(headers=resp.headers)
        fake_req._body = resp.content
        # Request.headers sets a _headers variable. We will need to set this
        # value to our fake request object.
        fake_req._headers = headers  # type: ignore (request._headers is property)

        return asyncio.run(api.output.from_http_request(fake_req))

    def close(self):
        self.client.close()
        return super().close()
