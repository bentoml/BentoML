from __future__ import annotations

import asyncio
import contextlib
import inspect
import io
import logging
import typing as t
from http import HTTPStatus
from urllib.parse import urljoin
from urllib.parse import urlparse

import anyio
import attr
from pydantic import RootModel

from bentoml._internal.utils.uri import uri_to_path
from bentoml.exceptions import BentoMLException

from ..types import File
from ..typing_utils import is_file_like
from ..typing_utils import is_image_type
from .base import AbstractClient

if t.TYPE_CHECKING:
    import yarl
    from aiohttp import ClientResponse
    from aiohttp import ClientSession
    from aiohttp import MultipartWriter

    from ..factory import Service
    from ..io_models import IODescriptor

    T = t.TypeVar("T", bound="HTTPClient")

logger = logging.getLogger("bentoml.io")
MAX_RETRIES = 3


@attr.define(slots=True)
class ClientEndpoint:
    name: str
    route: str
    doc: str | None = None
    input: dict[str, t.Any] = attr.field(factory=dict)
    output: dict[str, t.Any] = attr.field(factory=dict)
    input_spec: type[IODescriptor] | None = None
    output_spec: type[IODescriptor] | None = None
    stream_output: bool = False


@attr.define
class HTTPClient(AbstractClient):
    url: str
    endpoints: dict[str, ClientEndpoint] = attr.field(factory=dict)
    media_type: str = "application/json"
    timeout: int | None = None
    limiter: t.AsyncContextManager[t.Any] = contextlib.nullcontext()
    token: str | None = None
    _client: ClientSession | None = attr.field(init=False, default=None)
    _loop: asyncio.AbstractEventLoop | None = attr.field(init=False, default=None)

    def __init__(
        self,
        url: str,
        *,
        media_type: str = "application/json",
        service: Service[t.Any] | None = None,
        token: str | None = None,
    ) -> None:
        """Create a client instance from a URL.

        Args:
            url: The URL of the BentoML service.
            media_type: The media type to use for serialization. Defaults to
                "application/json".

        .. note::

            The client created with this method can only return primitive types without a model.
        """
        routes: dict[str, ClientEndpoint] = {}
        if service is None:
            import requests  # TODO: replace with httpx

            schema_url = urljoin(url, "/schema.json")
            resp = requests.get(schema_url)

            if not resp.ok:
                raise RuntimeError(f"Failed to fetch schema from {schema_url}")
            for route in resp.json()["routes"]:
                routes[route["name"]] = ClientEndpoint(
                    name=route["name"],
                    route=route["route"],
                    input=route["input"],
                    output=route["output"],
                    doc=route.get("doc"),
                    stream_output=route["output"].get("is_stream", False),
                )
        else:
            for name, method in service.apis.items():
                routes[name] = ClientEndpoint(
                    name=name,
                    route=method.route,
                    input=method.input_spec.model_json_schema(),
                    output=method.output_spec.model_json_schema(),
                    doc=method.doc,
                    input_spec=method.input_spec,
                    output_spec=method.output_spec,
                    stream_output=method.is_stream,
                )

        self.__attrs_init__(
            url=url, endpoints=routes, media_type=media_type, token=token
        )

    def __attrs_post_init__(self) -> None:
        from ..serde import ALL_SERDE

        self.serde = ALL_SERDE[self.media_type]()
        for name in self.endpoints:
            setattr(self, name, self._make_method(name))

    def _make_method(self, name: str) -> t.Callable[..., t.Any]:
        endpoint = self.endpoints[name]

        def method(*args: t.Any, **kwargs: t.Any) -> t.Any:
            return self.call(name, *args, **kwargs)

        method.__doc__ = endpoint.doc
        if endpoint.input_spec is not None:
            method.__annotations__ = endpoint.input_spec.__annotations__
            method.__signature__ = endpoint.input_spec.__signature__
        return method

    def _ensure_event_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    async def _get_client(self) -> ClientSession:
        import aiohttp
        from opentelemetry.instrumentation.aiohttp_client import create_trace_config

        from bentoml._internal.container import BentoMLContainer

        if self._client is None or self._client.closed:
            self._ensure_event_loop()
            if self._client is not None:
                await self._client.close()

            def strip_query_params(url: yarl.URL) -> str:
                return str(url.with_query(None))

            parsed = urlparse(self.url)
            client_kwargs = {
                "loop": self._loop,
                "trust_env": True,
                "trace_configs": [
                    create_trace_config(
                        # Remove all query params from the URL attribute on the span.
                        url_filter=strip_query_params,
                        tracer_provider=BentoMLContainer.tracer_provider.get(),
                    )
                ],
            }
            if self.timeout:
                client_kwargs["timeout"] = aiohttp.ClientTimeout(total=self.timeout)
            if parsed.scheme == "file":
                path = uri_to_path(self.url)
                conn = aiohttp.UnixConnector(
                    path=path,
                    loop=self._loop,
                    limit=800,  # TODO(jiang): make it configurable
                    keepalive_timeout=1800.0,
                )
                # base_url doesn't matter with UDS
                self._client = aiohttp.ClientSession(
                    base_url="http://127.0.0.1:8000", connector=conn, **client_kwargs
                )
            elif parsed.scheme == "tcp":
                url = f"http://{parsed.netloc}"
                self._client = aiohttp.ClientSession(url, **client_kwargs)
            else:
                self._client = aiohttp.ClientSession(self.url, **client_kwargs)
        return self._client

    async def _call(
        self,
        name: str,
        args: t.Sequence[t.Any],
        kwargs: dict[str, t.Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> t.Any:
        try:
            endpoint = self.endpoints[name]
        except KeyError:
            raise BentoMLException(f"Endpoint {name} not found") from None
        data = await self._prepare_request(endpoint, args, kwargs)
        resp = await self._request(endpoint.route, data, headers=headers)
        if endpoint.stream_output:
            return self._parse_stream_response(endpoint, resp)
        elif (
            endpoint.output.get("type") == "file"
            and self.media_type == "application/json"
        ):
            return await self._parse_file_response(endpoint, resp)
        else:
            return await self._parse_response(endpoint, resp)

    async def _request(
        self,
        url: str,
        data: bytes | MultipartWriter,
        headers: dict[str, str] | None = None,
    ) -> ClientResponse:
        from aiohttp import MultipartWriter
        from aiohttp.client_exceptions import ClientOSError

        from bentoml import __version__

        req_headers = {
            "Content-Type": f"multipart/form-data; boundary={data.boundary}"
            if isinstance(data, MultipartWriter)
            else self.media_type,
            "User-Agent": f"BentoML HTTP Client/{__version__}",
        }
        if self.token:
            req_headers["Authorization"] = f"Bearer {self.token}"
        if headers is not None:
            req_headers.update(headers)
        client = await self._get_client()
        async with self.limiter:
            for i in range(MAX_RETRIES):
                try:
                    resp = await client.post(url, data=data, headers=req_headers)
                except ClientOSError:
                    if i == MAX_RETRIES - 1:
                        raise
                    await self.close()
                    client = await self._get_client()
                else:
                    break
        if not resp.ok:
            raise BentoMLException(
                f"Error making request: {resp.status}: {await resp.text(errors='ignore')}",
                error_code=HTTPStatus(resp.status),
            )
        return resp

    def _build_multipart(
        self, model: IODescriptor | dict[str, t.Any]
    ) -> MultipartWriter | bytes:
        import aiohttp

        def is_file_field(k: str) -> bool:
            if isinstance(model, IODescriptor):
                return k in model.multipart_fields
            return (
                is_file_like(value := model[k])
                or isinstance(value, t.Sequence)
                and len(value) > 0
                and is_file_like(value[0])
            )

        if isinstance(model, dict):
            data = model
        else:
            data = {k: getattr(model, k) for k in model.model_fields}
        if self.media_type == "application/json":
            with aiohttp.MultipartWriter("form-data") as mp:
                for name, value in data.items():
                    if not is_file_field(name):
                        part = mp.append_json(value)
                        part.set_content_disposition("form-data", name=name)
                        continue
                    if not isinstance(value, t.Sequence):
                        value = [value]
                    for v in value:
                        if is_image_type(type(v)):
                            part = mp.append(
                                getattr(v, "_fp", v.fp),
                                headers={"Content-Type": f"image/{v.format.lower()}"},
                            )
                        else:
                            part = mp.append(v)
                        part.set_content_disposition(
                            "attachment", filename=part.filename, name=name
                        )
                return mp
        elif isinstance(model, dict):
            for k, v in data.items():
                if is_file_field(v) and not isinstance(v, File):
                    data[k] = File(v)
            return self.serde.serialize(data)
        else:
            return self.serde.serialize_model(model)

    async def _prepare_request(
        self,
        endpoint: ClientEndpoint,
        args: t.Sequence[t.Any],
        kwargs: dict[str, t.Any],
    ) -> bytes | MultipartWriter:
        if endpoint.input_spec is not None:
            model = endpoint.input_spec.from_inputs(*args, **kwargs)
            if not model.multipart_fields:
                return self.serde.serialize_model(model)
            else:
                return self._build_multipart(model)

        for name, value in zip(endpoint.input["properties"], args):
            if name in kwargs:
                raise TypeError(f"Duplicate argument {name}")
            kwargs[name] = value

        params = set(endpoint.input["properties"].keys())
        non_exist_args = set(kwargs.keys()) - set(params)
        if non_exist_args:
            raise TypeError(
                f"Arguments not found in endpoint {endpoint.name}: {non_exist_args}"
            )
        required = set(endpoint.input.get("required", []))
        missing_args = set(required) - set(kwargs.keys())
        if missing_args:
            raise TypeError(
                f"Missing required arguments in endpoint {endpoint.name}: {missing_args}"
            )
        multipart_fields = {
            k
            for k, v in kwargs.items()
            if is_file_like(v) or isinstance(v, t.Sequence) and v and is_file_like(v[0])
        }
        if multipart_fields:
            return self._build_multipart(kwargs)
        return self.serde.serialize(kwargs)

    def _deserialize_output(self, data: bytes, endpoint: ClientEndpoint) -> t.Any:
        if endpoint.output_spec is not None:
            model = self.serde.deserialize_model(data, endpoint.output_spec)
            if isinstance(model, RootModel):
                return model.root  # type: ignore
            return model
        elif (ot := endpoint.output.get("type")) == "string":
            return data.decode("utf-8")
        elif ot == "bytes":
            return data
        else:
            return self.serde.deserialize(data)

    async def _parse_response(
        self, endpoint: ClientEndpoint, resp: ClientResponse
    ) -> t.Any:
        data = await resp.read()
        return self._deserialize_output(data, endpoint)

    async def _parse_stream_response(
        self, endpoint: ClientEndpoint, resp: ClientResponse
    ) -> t.AsyncGenerator[t.Any, None]:
        buffer = bytearray()
        async for data, eoc in resp.content.iter_chunks():
            buffer.extend(data)
            if eoc:
                yield self._deserialize_output(bytes(buffer), endpoint)
                buffer.clear()

    async def _parse_file_response(
        self, endpoint: ClientEndpoint, resp: ClientResponse
    ) -> File:
        from ..types import Audio
        from ..types import Image
        from ..types import Video

        content_type = resp.headers.get("Content-Type")
        cls = File
        if content_type:
            if content_type.startswith("image/"):
                cls = Image
            elif content_type.startswith("audio/"):
                cls = Audio
            elif content_type.startswith("video/"):
                cls = Video
        return cls(io.BytesIO(await resp.read()), media_type=content_type)

    async def close(self) -> None:
        if self._client is not None and not self._client.closed:
            await self._client.close()


class SyncHTTPClient(HTTPClient):
    """A synchronous client for BentoML service.

    Example:

        with SyncHTTPClient("http://localhost:3000") as client:
            resp = client.call("classify", input_series=[[1,2,3,4]])
            assert resp == [0]
            # Or using named method directly
            resp = client.classify(input_series=[[1,2,3,4]])
            assert resp == [0]
    """

    def call(self, __name: str, /, *args: t.Any, **kwargs: t.Any) -> t.Any:
        from bentoml._internal.utils import async_gen_to_sync

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        res = loop.run_until_complete(self._call(__name, args, kwargs))
        if inspect.isasyncgen(res):
            return async_gen_to_sync(res, loop=loop)
        return res

    def __enter__(self) -> HTTPClient:
        return self

    def __exit__(self, exc_type: t.Any, exc: t.Any, tb: t.Any) -> None:
        return anyio.run(self.close)

    def is_ready(
        self, timeout: int | None = None, headers: dict[str, str] | None = None
    ) -> bool:
        return anyio.run(AsyncHTTPClient.is_ready, self, timeout, headers)


class AsyncHTTPClient(HTTPClient):
    """An asynchronous client for BentoML service.

    Example:

        async with AsyncHTTPClient("http://localhost:3000") as client:
            resp = await client.call("classify", input_series=[[1,2,3,4]])
            assert resp == [0]
            # Or using named method directly
            resp = await client.classify(input_series=[[1,2,3,4]])
            assert resp == [0]

    .. note::

        If the endpoint returns an async generator, it should be awaited before iterating.

        Example:

            resp = await client.stream(prompt="hello")
            async for data in resp:
                print(data)
    """

    async def is_ready(
        self, timeout: int | None = None, headers: dict[str, str] | None = None
    ) -> bool:
        import aiohttp

        client = await self._get_client()
        request_params: dict[str, t.Any] = {"headers": headers}
        if timeout is not None:
            request_params["timeout"] = aiohttp.ClientTimeout(total=timeout)
        try:
            async with client.get("/readyz", **request_params) as resp:
                return resp.status == 200
        except asyncio.TimeoutError:
            logger.warn("Timed out waiting for runner to be ready")
            return False

    def call(self, __name: str, /, *args: t.Any, **kwargs: t.Any) -> t.Any:
        try:
            endpoint = self.endpoints[__name]
        except KeyError:
            raise BentoMLException(f"Endpoint {__name} not found") from None
        if endpoint.stream_output:
            return self._get_stream(endpoint, *args, **kwargs)
        else:
            return self._call(__name, args, kwargs)

    async def _get_stream(
        self, endpoint: ClientEndpoint, *args: t.Any, **kwargs: t.Any
    ) -> t.AsyncGenerator[t.Any, None]:
        resp = await self._call(endpoint.name, args, kwargs)
        assert inspect.isasyncgen(resp)
        async for data in resp:
            yield data

    async def __aenter__(self: T) -> T:
        return self

    async def __aexit__(self, *args: t.Any) -> None:
        return await self.close()
