from __future__ import annotations

import json
import time
import socket
import typing as t
import logging
import urllib.error
import urllib.parse
import urllib.request
from http.client import HTTPConnection
from urllib.parse import urlparse

import aiohttp
import requests
import multidict
import starlette.requests
import requests.structures
import starlette.datastructures

from . import BaseSyncClient
from . import BaseAsyncClient
from . import ensure_exec_coro
from .. import io_descriptors
from ..service import Service
from ...exceptions import RemoteException
from ...exceptions import BentoMLException
from ..configuration import get_debug_mode
from ..service.openapi import APP_TAG
from ..service.inference_api import InferenceAPI

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from starlette.responses import Response

    from . import Client
    from . import ClientProtocol


class HTTPClientMixin:
    @staticmethod
    def wait_until_server_ready(
        host: str,
        port: int,
        timeout: float = 30,
        *,
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

    @staticmethod
    def _process_call_kwargs(
        **kwargs: t.Any,
    ) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
        """Helpers mixin to process incoming kwargs for call method.
        It will pop out all kwargs starts with `_grpc_`.
        All kwargs prefix with `_http_` will be then parsed to coresponding Session.
        The rest will then be pass to the IODescriptor.
        """
        # All gRPC kwargs should be poped out.
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_grpc_")}

        # Check for all kwargs that can be parsed to ClientSession.
        # Currently, we only support parsing headers.
        _request_kwargs = {
            k[6:]: v for k, v in kwargs.items() if k.startswith("_http_")
        }
        io_kwargs = {k: v for k, v in kwargs.items() if k[6:] not in _request_kwargs}

        return _request_kwargs, io_kwargs

    @staticmethod
    def _process_requests_kwargs(
        request_kwargs: dict[str, t.Any], resp: Response
    ) -> tuple[dict[str, t.Any], multidict.MultiDict[t.Any]]:
        """Parsing given request_kwargs and extract headers from the 'fake_resp'
        then returns the headers and the rest of the request_kwargs.

        Note that if users specify `content-type`, it will override the default content-type
        from the 'fake_resp'

        It will also pop out the `data` from the request_kwargs, since the client will manage that.
        """
        # Returns the processed request_kwargs and headers
        headers = multidict.MultiDict({"Content-Type": resp.headers["content-type"]})
        if "headers" in request_kwargs:
            _headers = multidict.MultiDict(request_kwargs.pop("headers") or {})
            if "content-type" in _headers:
                logger.warning(
                    "Overwritting default content-type header '%s' with '%s'",
                    resp.headers["content-type"],
                    _headers["content-type"],
                )
                headers["Content-Type"] = _headers.pop("content-type")
            # update the rest with user provided headers
            headers.update(_headers)
        if "data" in request_kwargs:
            logger.warning("'data' is passed to call, which will be ignored.")
            request_kwargs.pop("data")

        logger.debug("Request headers: %s", headers)
        if request_kwargs:
            logger.debug("Request arguments: %s", request_kwargs)

        return request_kwargs, headers

    @staticmethod
    def _prepare_base_class(
        klass: type[Client], server_url: str, **_: t.Any
    ) -> ClientProtocol:
        """Shared logics for `from_url` that can be used for both Async and Sync implementation."""
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
        endpoint_kwds_map: dict[str, list[str]] = {}

        for route, spec in openapi_spec["paths"].items():
            for meth_spec in spec.values():
                if "tags" in meth_spec and APP_TAG.name in meth_spec["tags"]:
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
                    if "x-bentoml-func-kwds" in meth_spec:
                        endpoint_kwds_map[meth_spec["x-bentoml-name"]] = meth_spec[
                            "x-bentoml-func-kwds"
                        ]
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

        HttpClient = klass(dummy_service, server_url)
        supports_kwds_assignment = len(endpoint_kwds_map) > 0
        HttpClient.supports_kwds_assignment = supports_kwds_assignment
        if supports_kwds_assignment:
            HttpClient._endpoint_kwds_map = endpoint_kwds_map
        return HttpClient


class _Session(requests.Session):
    """Default requests.Session plus support for base_url."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        super(_Session, self).__init__()

    def request(
        self, method: str | bytes, url: str | bytes, *args: t.Any, **kwargs: t.Any
    ) -> t.Any:
        url = urllib.parse.urljoin(self.base_url, str(url))
        return super(_Session, self).request(method, url, *args, **kwargs)


class HTTPClient(BaseSyncClient, HTTPClientMixin):
    _conn_type: _Session

    async def async_health(self) -> t.Any:
        logger.warning(
            "Calling 'async_health' from HTTPClient is now deprecated. Create a AsyncHTTPClient and use 'health' instead."
        )
        async with aiohttp.ClientSession(self.server_url) as sess:
            async with sess.get("/readyz") as resp:
                return resp

    def health(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        assert self._conn_type, "Session not found (error during client initialisation)"
        return self._conn_type.get("/readyz")

    @classmethod
    def from_url(cls, server_url: str, **kwargs: t.Any) -> HTTPClient:
        HttpClient = cls._prepare_base_class(cls, server_url, **kwargs)
        HttpClient._conn_type = _Session(HttpClient.server_url)
        return t.cast("HTTPClient", HttpClient)

    def request(self, *args: t.Any, **kwargs: t.Any):
        """
        Expose the internal ``requests.Session.request`` method. Note that the url
        argument should be the path to the given endpoint, not the full url as the client
        will construct the full url based on the server_url.

        .. code-block:: python

            client.request("GET", "/api/v1/metadata")

        Returns:

            ``requests.Response``: Refer to requests documentation for more information.

        .. note::

            Should only be used when you need to send some requests other than
            the defined APIs endpoints to the server
        """
        if not self._conn_type:
            raise RuntimeError("Client is not created correctly.")
        return self._conn_type.request(*args, **kwargs)

    def _sync_call(
        self, inp: t.Any = None, *, _bentoml_api: InferenceAPI, **kwargs: t.Any
    ):
        request_kwargs, io_kwargs = self._process_call_kwargs(**kwargs)

        fake_resp = ensure_exec_coro(
            _bentoml_api.input.to_http_response(
                self._prepare_call_inputs(
                    inp=inp, io_kwargs=io_kwargs, api=_bentoml_api
                ),
                None,
            )
        )
        req_kwargs, headers = self._process_requests_kwargs(request_kwargs, fake_resp)
        try:
            with self._conn_type.post(
                "/" + _bentoml_api.route
                if not _bentoml_api.route.startswith("/")
                else _bentoml_api.route,
                data=fake_resp.body,
                headers=dict(headers),
                **req_kwargs,
            ) as resp:
                if resp.status_code != 200:
                    raise BentoMLException(
                        f"Error making request [status={resp.status_code}]: {str(resp.content.decode())}"
                    )

                fake_req = starlette.requests.Request(scope={"type": "http"})
                headers = starlette.datastructures.Headers(headers=resp.headers)
                fake_req._body = resp.content
                fake_req._headers = headers
        except Exception as e:
            logger.error(
                "Exception caught while making inference request to %s:\n",
                _bentoml_api.route,
            )
            logger.error(e)
            raise

        return ensure_exec_coro(_bentoml_api.output.from_http_request(fake_req))

    # XXX: This method is mainly for backward compatible.
    async def _call(
        self, inp: t.Any = None, *, _bentoml_api: InferenceAPI, **kwargs: t.Any
    ) -> t.Any:
        logger.warning(
            "Calling 'async_%s' from HTTPClient is now deprecated. Create a AsyncHTTPClient and use '%s' instead.",
            _bentoml_api.name,
            _bentoml_api.name,
        )
        _request_kwargs, io_kwargs = self._process_call_kwargs(**kwargs)
        fake_resp = await _bentoml_api.input.to_http_response(
            self._prepare_call_inputs(inp=inp, io_kwargs=io_kwargs, api=_bentoml_api),
            None,
        )
        req_kwargs, headers = self._process_requests_kwargs(_request_kwargs, fake_resp)
        try:
            async with aiohttp.ClientSession(self.server_url) as sess:
                async with sess.post(
                    "/" + _bentoml_api.route
                    if not _bentoml_api.route.startswith("/")
                    else _bentoml_api.route,
                    data=fake_resp.body,
                    headers=headers,
                    **req_kwargs,
                ) as resp:
                    if resp.status != 200:
                        raise BentoMLException(
                            f"Error making request: {resp.status}: {str(await resp.read())}"
                        )

                    fake_req = starlette.requests.Request(scope={"type": "http"})
                    headers = starlette.datastructures.Headers(headers=resp.headers)
                    fake_req._body = await resp.read()
                    # Request.headers sets a _headers variable. We will need to set this
                    # value to our fake request object.
                    fake_req._headers = headers  # type: ignore (request._headers is property)
        except Exception as e:
            logger.error(
                "Exception caught while making inference request to %s:\n",
                _bentoml_api.route,
            )
            logger.error(e)
            raise

        return await _bentoml_api.output.from_http_request(fake_req)


class AsyncHTTPClient(BaseAsyncClient, HTTPClientMixin):
    _conn_type: aiohttp.ClientSession

    async def health(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return await self._conn_type.get("/healthz")

    @classmethod
    def from_url(cls, server_url: str, **kwargs: t.Any) -> AsyncHTTPClient:
        HttpClient = cls._prepare_base_class(cls, server_url, **kwargs)
        HttpClient._conn_type = aiohttp.ClientSession(HttpClient.server_url)
        return t.cast("AsyncHTTPClient", HttpClient)

    async def request(self, *args: t.Any, **kwargs: t.Any):
        """
        Expose the internal ``aiohttp.ClientSession.request`` method.

        Reference: https://docs.aiohttp.org/en/stable/client_reference.html#aiohttp.ClientSession.request

        Returns:

            ``aiohttp.ClientResponse``

        .. note::

            Should only be used when you need to send some requests other than
            the defined APIs endpoints to the server.
        """
        if not self._conn_type:
            raise RuntimeError("Client is not created correctly.")
        return await self._conn_type.request(*args, **kwargs)

    async def _call(
        self, inp: t.Any = None, *, _bentoml_api: InferenceAPI, **kwargs: t.Any
    ) -> t.Any:
        _request_kwargs, io_kwargs = self._process_call_kwargs(**kwargs)
        fake_resp = await _bentoml_api.input.to_http_response(
            self._prepare_call_inputs(inp=inp, io_kwargs=io_kwargs, api=_bentoml_api),
            None,
        )
        req_kwargs, headers = self._process_requests_kwargs(_request_kwargs, fake_resp)
        try:
            resp = await self._conn_type.post(
                "/" + _bentoml_api.route
                if not _bentoml_api.route.startswith("/")
                else _bentoml_api.route,
                data=fake_resp.body,
                headers=headers,
                **req_kwargs,
            )
            if resp.status != 200:
                raise BentoMLException(
                    f"Error making request: {resp.status}: {str(await resp.read())}"
                )

            fake_req = starlette.requests.Request(scope={"type": "http"})
            headers = starlette.datastructures.Headers(headers=resp.headers)
            fake_req._body = await resp.read()
            # Request.headers sets a _headers variable. We will need to set this
            # value to our fake request object.
            fake_req._headers = headers
        except Exception as e:
            logger.error(
                "Exception caught while making inference request to %s:\n",
                _bentoml_api.route,
            )
            logger.error(e)
            raise

        return await _bentoml_api.output.from_http_request(fake_req)
