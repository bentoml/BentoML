from __future__ import annotations

import json
import typing as t
import asyncio
import functools
from abc import ABC
from abc import abstractmethod
from http.client import HTTPConnection
from urllib.parse import urlparse

import aiohttp
import starlette.requests
import starlette.datastructures

import bentoml
from bentoml import Service

from .exceptions import BentoMLException
from ._internal.service.inference_api import InferenceAPI


class Client(ABC):
    server_url: str

    def __init__(self, svc: Service, server_url: str):
        self._svc = svc
        self.server_url = server_url
        if len(self._svc.apis) == 0:
            raise BentoMLException("No APIs were found when constructing client")

        for name, api in self._svc.apis.items():
            if not hasattr(self, name):
                setattr(
                    self, name, functools.partial(self._sync_call, _bentoml_api=api)
                )

        for name, api in self._svc.apis.items():
            if not hasattr(self, f"async_{name}"):
                setattr(
                    self,
                    f"async_{name}",
                    functools.partial(self._call, _bentoml_api=api),
                )

    def call(self, bentoml_api_name: str, inp: t.Any = None, **kwargs: t.Any) -> t.Any:
        return asyncio.run(self.async_call(bentoml_api_name, inp))

    async def async_call(
        self, bentoml_api_name: str, inp: t.Any = None, **kwargs: t.Any
    ) -> t.Any:
        return await self._call(inp, _bentoml_api=self._svc.apis[bentoml_api_name])

    def _sync_call(
        self, inp: t.Any = None, *, _bentoml_api: InferenceAPI, **kwagrs: t.Any
    ):
        return asyncio.run(self._call(inp, _bentoml_api=_bentoml_api))

    @abstractmethod
    async def _call(
        self, inp: t.Any = None, *, _bentoml_api: InferenceAPI, **kwargs: t.Any
    ) -> t.Any:
        raise NotImplementedError

    @staticmethod
    def from_url(server_url: str) -> Client:
        server_url = server_url if "://" in server_url else "http://" + server_url
        url_parts = urlparse(server_url)

        # TODO: SSL and grpc support
        conn = HTTPConnection(url_parts.netloc)
        conn.request("GET", "/docs.json")
        resp = conn.getresponse()
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
                    dummy_service.apis[meth_spec["x-bentoml-name"]] = InferenceAPI(
                        None,
                        bentoml.io.from_spec(
                            meth_spec["requestBody"]["x-bentoml-io-descriptor"]
                        ),
                        bentoml.io.from_spec(
                            meth_spec["responses"]["200"]["x-bentoml-io-descriptor"]
                        ),
                        name=meth_spec["x-bentoml-name"],
                        doc=meth_spec["description"],
                        route=route.lstrip("/"),
                    )

        res = HTTPClient(dummy_service, server_url)
        res.server_url = server_url
        return res


class HTTPClient(Client):
    _svc: Service

    async def _call(
        self, inp: t.Any = None, *, _bentoml_api: InferenceAPI, **kwargs: t.Any
    ) -> t.Any:
        api = _bentoml_api

        if api.multi_input:
            if inp is not None:
                raise BentoMLException(
                    f"'{api.name}' takes multiple inputs; all inputs must be passed as keyword arguments."
                )
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
