from __future__ import annotations

import json
import time
import socket
import typing as t
import logging
import urllib.error
import urllib.request
from http.client import HTTPConnection
from urllib.parse import urlparse

import aiohttp
import starlette.requests
import starlette.datastructures

from . import Client
from .. import io_descriptors as io
from ..service import Service
from ...exceptions import RemoteException
from ...exceptions import BentoMLException
from ..configuration import get_debug_mode
from ..service.inference_api import InferenceAPI

logger = logging.getLogger(__name__)


class HTTPClient(Client):
    def wait_until_server_ready(
        self,
        *,
        server_url: str | None = None,
        timeout: int = 30,
        check_interval: int = 1,
        # set kwargs here to omit gRPC kwargs
        **kwargs: t.Any,
    ) -> None:
        start_time = time.time()
        if server_url is None:
            server_url = self.server_url

        proxy_handler = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(proxy_handler)
        logger.debug("Waiting for host %s to be ready.", server_url)
        while time.time() - start_time < timeout:
            try:
                if opener.open(f"http://{server_url}/readyz", timeout=1).status == 200:
                    break
                else:
                    time.sleep(check_interval)
            except (ConnectionError, urllib.error.URLError, socket.timeout) as err:
                logger.debug("[%s] Retrying to connect to the host %s", err, server_url)
                time.sleep(check_interval)
        raise TimeoutError(
            f"Timed out waiting {timeout} seconds for server at '{server_url}' to be ready."
        )

    @classmethod
    def from_url(cls, server_url: str, **kwargs: t.Any) -> HTTPClient:
        server_url = server_url if "://" in server_url else "http://" + server_url
        url_parts = urlparse(server_url)

        # TODO: SSL and grpc support
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
                if "Service APIs" in meth_spec["tags"]:
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
                            "Failed to instantiate client for API %s: ",
                            meth_spec["x-bentoml-name"],
                            e,
                        )

        return cls(dummy_service, server_url)

    async def _call(
        self, inp: t.Any = None, *, _bentoml_api: InferenceAPI, **kwargs: t.Any
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
        req_body = fake_resp.body

        async with aiohttp.ClientSession(self.server_url) as sess:
            async with sess.post(
                "/" + api.route,
                data=req_body,
                headers={"content-type": fake_resp.headers["content-type"]},
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

        return await api.output.from_http_request(fake_req)
