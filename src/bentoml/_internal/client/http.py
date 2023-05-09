from __future__ import annotations

import json
import time
import socket
import typing as t
import asyncio
import logging
import urllib.error
import urllib.request
from http.client import HTTPConnection
from urllib.parse import urlparse

import aiohttp
import multidict
import starlette.requests
import starlette.datastructures

from . import Client
from .. import io_descriptors
from ..service import Service
from ...exceptions import RemoteException
from ...exceptions import BentoMLException
from ..configuration import get_debug_mode
from ..service.inference_api import InferenceAPI

logger = logging.getLogger(__name__)


class HTTPClient(Client):
    _session: aiohttp.ClientSession

    @staticmethod
    def wait_until_server_ready(
        host: str,
        port: int,
        timeout: float = 30,
        check_interval: int = 1,
        # set kwargs here to omit gRPC kwargs
        **kwargs: t.Any,
    ) -> None:
        start_time = time.time()
        status = None

        logger.debug("Waiting for host %s to be ready.", f"{host}:{port}")
        while time.time() - start_time < timeout:
            try:
                conn = HTTPConnection(host, port)
                conn.request("GET", "/readyz")
                status = conn.getresponse().status
                if status == 200:
                    break
                else:
                    time.sleep(check_interval)
            except (
                ConnectionError,
                urllib.error.URLError,
                socket.timeout,
                ConnectionRefusedError,
            ):
                logger.debug("Server is not ready. Retrying...")
                time.sleep(check_interval)

        # try to connect one more time and raise exception.
        try:
            conn = HTTPConnection(host, port)
            conn.request("GET", "/readyz")
            status = conn.getresponse().status
            if status != 200:
                raise TimeoutError(
                    f"Timed out waiting {timeout} seconds for server at '{host}:{port}' to be ready."
                )
        except (
            ConnectionError,
            urllib.error.URLError,
            socket.timeout,
            ConnectionRefusedError,
            TimeoutError,
        ) as err:
            logger.error("Timed out while connecting to %s:%s:", host, port)
            logger.error(err)
            raise

    async def async_health(self) -> t.Any:
        return await self._session.get("/healthz")

    def health(self) -> t.Any:
        return self._ensure_exec_coro(self.async_health())

    @classmethod
    def from_url(cls, server_url: str, **kwargs: t.Any) -> HTTPClient:
        server_url = server_url if "://" in server_url else "http://" + server_url
        url_parts = urlparse(server_url)

        # TODO: SSL support
        conn = HTTPConnection(url_parts.netloc)
        conn.set_debuglevel(logging.DEBUG if get_debug_mode() else 0)
        conn.request("GET", url_parts.path + "/docs.json")
        resp = conn.getresponse()
        if resp.status != 200:
            raise RemoteException(
                f"Failed to get OpenAPI schema from the server: {resp.status} {resp.reason}:\n{resp.read()}"
            )
        openapi_spec = json.load(resp)
        conn.close()

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
                        api = InferenceAPI(
                            None,
                            io_descriptors.from_spec(
                                meth_spec["requestBody"]["x-bentoml-io-descriptor"]
                            ),
                            io_descriptors.from_spec(
                                meth_spec["responses"]["200"]["x-bentoml-io-descriptor"]
                            ),
                            name=meth_spec["x-bentoml-name"],
                            doc=meth_spec["description"],
                            route=route.lstrip("/"),
                        )
                        dummy_service.apis[meth_spec["x-bentoml-name"]] = api
                    except BentoMLException as e:
                        logger.error(
                            "Failed to instantiate client for API %s: ",
                            meth_spec["x-bentoml-name"],
                            e,
                        )

        HttpClient = cls(dummy_service, server_url)
        HttpClient._session = aiohttp.ClientSession(HttpClient.server_url)
        return HttpClient

    async def async_request(self, *args: t.Any, **kwargs: t.Any):
        """
        Expose the internal ``aiohttp.ClientSession.request`` method.

        Returns:

            ``aiohttp.ClientResponse``

        .. note::

            Should only be used when you need to send some requests other than
            the defined APIs endpoints to the server
        """
        if not self._session:
            raise RuntimeError("Client is not created correctly.")
        return await self._session.request(*args, **kwargs)

    @staticmethod
    def _ensure_exec_coro(coro: t.Coroutine[t.Any, t.Any, t.Any]) -> t.Any:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result()
        else:
            return loop.run_until_complete(coro)

    def request(self, *args: t.Any, **kwargs: t.Any):
        """
        Sync version of ``async_request``.
        This exposes the underlying ``aiohttp.ClientSession.request`` method.
        """
        # NOTE: We can't use asyncio.run here because timer context for
        # self._session.request must be used inside a task.
        return self._ensure_exec_coro(self.async_request(*args, **kwargs))

    def __del__(self):
        # Close connection when this object is destroyed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._session.close())
            else:
                loop.run_until_complete(self._session.close())
        except Exception as e:
            logger.error("Exception caught while gc the client object:\n")
            logger.error(e)
            raise

    def _sync_call(
        self, inp: t.Any = None, *, _bentoml_api: InferenceAPI, **kwargs: t.Any
    ):
        # NOTE: We need a check here so that client.call()
        # shouldn't be called inside a async context, since _sync_call
        # will actually create a new event loop via asyncio.run
        # Chances are if you are already inside a async context, then you should
        # use async_call instead.
        if asyncio.get_event_loop().is_running():
            raise RuntimeError(
                "Cannot call sync_call inside a async context. Since you are already inside a async context, consider using 'async_call' or 'async_<api>' instead."
            )
        return self._ensure_exec_coro(
            self._call(inp, _bentoml_api=_bentoml_api, **kwargs)
        )

    async def _call(
        self, inp: t.Any = None, *, _bentoml_api: InferenceAPI, **kwargs: t.Any
    ) -> t.Any:
        assert self._session, "Client is not created correctly."
        # All gRPC kwargs should be poped out.
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_grpc_")}

        # Check for all kwargs that can be parsed to ClientSession.
        # Currently, we only support parsing headers.
        _request_kwargs = {
            k[6:]: v for k, v in kwargs.items() if k.startswith("_http_")
        }
        multi_input_kwargs = {
            k: v for k, v in kwargs.items() if not k.startswith("_http_")
        }

        api = _bentoml_api
        if api.multi_input:
            if inp is not None:
                raise BentoMLException(
                    f"'{api.name}' takes multiple inputs; all inputs must be passed as keyword arguments."
                )
            fake_resp = await api.input.to_http_response(multi_input_kwargs, None)
        else:
            fake_resp = await api.input.to_http_response(inp, None)
        req_body = fake_resp.body

        headers = multidict.MultiDict(
            {"Content-Type": fake_resp.headers["content-type"]}
        )
        if "headers" in _request_kwargs:
            _headers = multidict.MultiDict(_request_kwargs.pop("headers") or {})
            if "content-type" in _headers:
                logger.warning(
                    "Overwritting default content-type header %s with %s",
                    fake_resp.headers["content-type"],
                    _headers["content-type"],
                )
                headers["Content-Type"] = _headers.pop("content-type")
            # update the rest with user provided headers
            headers.update(_headers)
        if "data" in _request_kwargs:
            logger.warning("'data' is passed to call, which will be ignored.")
            _request_kwargs.pop("data")
        if "json" in _request_kwargs:
            logger.warning("'json' is passed to call, which will be ignored.")
            _request_kwargs.pop("json")

        logger.debug("Sending request to %s with headers %s", api.route, headers)
        if _request_kwargs:
            logger.debug("Request arguments: %s", _request_kwargs)

        try:
            async with self._session.post(
                "/" + api.route if not api.route.startswith("/") else api.route,
                data=req_body,
                headers=headers,
                **_request_kwargs,
            ) as resp:
                if resp.status != 200:
                    res = await resp.read()
                    raise BentoMLException(
                        f"Error making request [status={resp.status}]: {str(res.decode())}"
                    )

                fake_req = starlette.requests.Request(scope={"type": "http"})
                headers = starlette.datastructures.Headers(headers=resp.headers)
                fake_req._body = await resp.read()
                # Request.headers sets a _headers variable. We will need to set this
                # value to our fake request object.
                fake_req._headers = headers  # type: ignore (request._headers is property)
        except Exception as e:
            logger.error(
                "Exception caught while making inference request to %s:\n", api.route
            )
            logger.error(e)
            raise

        return await api.output.from_http_request(fake_req)
