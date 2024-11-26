from __future__ import annotations

import inspect
import io
import json
import logging
import mimetypes
import os
import pathlib
import tempfile
import time
import typing as t
from abc import abstractmethod
from functools import cached_property
from http import HTTPStatus
from urllib.parse import urljoin
from urllib.parse import urlparse

import attr
import httpx

from _bentoml_sdk import IODescriptor
from _bentoml_sdk.typing_utils import is_image_type
from bentoml import __version__
from bentoml._internal.utils.uri import uri_to_path
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import NotFound
from bentoml.exceptions import ServiceUnavailable

from ..serde import Payload
from .base import AbstractClient
from .base import ClientEndpoint
from .base import map_exception
from .task import AsyncTask
from .task import Task

if t.TYPE_CHECKING:
    from httpx._types import RequestFiles
    from PIL import Image

    from _bentoml_sdk import Service
    from bentoml._internal.external_typing import ASGIApp

    from ..serde import Serde

    T = t.TypeVar("T", bound="HTTPClient[t.Any]")
    A = t.TypeVar("A")
C = t.TypeVar("C", httpx.Client, httpx.AsyncClient)
AnyClient = t.TypeVar("AnyClient", httpx.Client, httpx.AsyncClient)
logger = logging.getLogger("bentoml.io")
MAX_RETRIES = 3


def is_http_url(url: str) -> bool:
    return urlparse(url).scheme in {"http", "https"}


def to_async_iterable(iterable: t.Iterable[A]) -> t.AsyncIterable[A]:
    async def _gen() -> t.AsyncIterator[A]:
        for item in iterable:
            yield item

    return _gen()


@attr.define
class HTTPClient(AbstractClient, t.Generic[C]):
    client_cls: t.ClassVar[type[httpx.Client] | type[httpx.AsyncClient]]

    url: str
    endpoints: dict[str, ClientEndpoint] = attr.field(factory=dict)
    media_type: str = "application/json"
    timeout: float = 30
    default_headers: dict[str, str] = attr.field(factory=dict)
    app: ASGIApp | None = None

    _opened_files: list[io.BufferedReader] = attr.field(init=False, factory=list)
    _temp_dir: tempfile.TemporaryDirectory[str] = attr.field(init=False)

    @staticmethod
    def _make_client(
        client_cls: type[AnyClient],
        url: str,
        headers: t.Mapping[str, str],
        timeout: float,
        app: ASGIApp | None = None,
    ) -> AnyClient:
        parsed = urlparse(url)
        transport = None
        if parsed.scheme == "file":
            uds = uri_to_path(url)
            if client_cls is httpx.Client:
                transport = httpx.HTTPTransport(uds=uds)
            else:
                transport = httpx.AsyncHTTPTransport(uds=uds)
            url = "http://127.0.0.1:3000"
        elif parsed.scheme == "tcp":
            url = f"http://{parsed.netloc}"
        elif app is not None:
            if client_cls is httpx.Client:
                from a2wsgi import ASGIMiddleware

                transport = httpx.WSGITransport(app=ASGIMiddleware(app))
            else:
                transport = httpx.ASGITransport(app=app)
        return client_cls(
            base_url=url,
            transport=transport,  # type: ignore
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
        )

    @_temp_dir.default  # type: ignore
    def default_temp_dir(self) -> tempfile.TemporaryDirectory[str]:
        return tempfile.TemporaryDirectory(prefix="bentoml-client-")

    def __init__(
        self,
        url: str,
        *,
        media_type: str = "application/json",
        service: Service[t.Any] | None = None,
        server_ready_timeout: float | None = None,
        token: str | None = None,
        timeout: float = 30,
        app: ASGIApp | None = None,
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
        default_headers = {"User-Agent": f"BentoML HTTP Client/{__version__}"}
        if token is None:
            token = os.getenv("BENTO_CLOUD_API_KEY")
        if token:
            default_headers["Authorization"] = f"Bearer {token}"

        if service is not None:
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
                    is_task=method.is_task,
                )

            from bentoml._internal.context import server_context

            default_headers.update(
                {
                    "Bento-Name": server_context.bento_name,
                    "Bento-Version": server_context.bento_version,
                    "Runner-Name": service.name,
                    "Yatai-Bento-Deployment-Name": server_context.yatai_bento_deployment_name,
                    "Yatai-Bento-Deployment-Namespace": server_context.yatai_bento_deployment_namespace,
                }
            )
        self.__attrs_init__(  # type: ignore
            url=url,
            endpoints=routes,
            media_type=media_type,
            default_headers=default_headers,
            timeout=timeout,
            app=app,
        )
        if app is None and (server_ready_timeout is None or server_ready_timeout > 0):
            self.wait_until_server_ready(server_ready_timeout)
        if service is None:
            schema_url = urljoin(url, "/schema.json")

            with self._make_client(
                httpx.Client, url, default_headers, timeout, app=app
            ) as client:
                resp = client.get("/schema.json")

                if resp.is_error:
                    raise BentoMLException(f"Failed to fetch schema from {schema_url}")
                for route in resp.json()["routes"]:
                    self.endpoints[route["name"]] = ClientEndpoint(
                        name=route["name"],
                        route=route["route"],
                        input=route["input"],
                        output=route["output"],
                        doc=route.get("doc"),
                        stream_output=route["output"].get("is_stream", False),
                        is_task=route.get("is_task", False),
                    )
        super().__init__()

    @cached_property
    def client(self) -> C:
        return self._make_client(
            self.client_cls, self.url, self.default_headers, self.timeout, self.app
        )

    @cached_property
    def serde(self) -> Serde:
        from ..serde import ALL_SERDE

        return ALL_SERDE[self.media_type]()

    def _build_request(
        self,
        endpoint: ClientEndpoint,
        args: t.Sequence[t.Any],
        kwargs: dict[str, t.Any],
        headers: t.Mapping[str, str],
    ) -> httpx.Request:
        from opentelemetry import propagate

        headers = httpx.Headers({"Content-Type": self.media_type, **headers})
        propagate.inject(headers)
        if endpoint.input_spec is not None:
            model = endpoint.input_spec.from_inputs(*args, **kwargs)
            if model.multipart_fields and self.media_type == "application/json":
                return self._build_multipart(endpoint, model, headers)
            else:
                payload = self.serde.serialize_model(model)
                headers.update(payload.headers)
                return self.client.build_request(
                    "POST",
                    endpoint.route,
                    headers=headers,
                    content=to_async_iterable(payload.data)
                    if self.client_cls is httpx.AsyncClient
                    else payload.data,
                )

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
        has_file = any(
            schema.get("type") == "file"
            or schema.get("type") == "array"
            and schema["items"].get("type") == "file"
            for schema in endpoint.input["properties"].values()
        )
        if has_file and self.media_type == "application/json":
            return self._build_multipart(endpoint, kwargs, headers)
        payload = self.serde.serialize(kwargs, endpoint.input)
        headers.update(payload.headers)
        return self.client.build_request(
            "POST",
            endpoint.route,
            content=to_async_iterable(payload.data)
            if self.client_cls is httpx.AsyncClient
            else payload.data,
            headers=headers,
        )

    def wait_until_server_ready(self, timeout: int | None = None) -> None:
        if timeout is None:
            timeout = self.timeout
        with self._make_client(
            httpx.Client, self.url, self.default_headers, timeout
        ) as client:
            start = time.monotonic()
            while time.monotonic() - start < timeout:
                try:
                    resp = client.get("/readyz")
                    if resp.status_code == 200:
                        return
                except (httpx.TimeoutException, httpx.ConnectError):
                    pass
        raise ServiceUnavailable(f"Server is not ready after {timeout} seconds")

    def _build_multipart(
        self,
        endpoint: ClientEndpoint,
        model: IODescriptor | dict[str, t.Any],
        headers: httpx.Headers,
    ) -> httpx.Request:
        def is_file_field(k: str) -> bool:
            if isinstance(model, IODescriptor):
                return k in model.multipart_fields
            if (f := endpoint.input["properties"].get(k, {})).get("type") == "file":
                return True
            if f.get("type") == "array" and f["items"].get("type") == "file":
                return True
            return False

        if isinstance(model, dict):
            fields = model
        else:
            fields = {k: getattr(model, k) for k in model.model_fields}
        data: dict[str, t.Any] = {}
        files: RequestFiles = []

        for name, value in fields.items():
            if not is_file_field(name):
                data[name] = json.dumps(value)
                continue
            if not isinstance(value, (list, tuple)):
                value = [value]

            for v in value:
                if isinstance(v, str) and not is_http_url(v):
                    v = pathlib.Path(v)
                if is_image_type(type(v)):
                    fp = getattr(v, "_fp", v.fp)
                    fname = getattr(fp, "name", None)
                    fmt = v.format.lower()
                    files.append(
                        (
                            name,
                            (
                                pathlib.Path(fname).name
                                if fname
                                else f"upload-image.{fmt}",
                                fp,
                                f"image/{fmt}",
                            ),
                        )
                    )
                elif isinstance(v, pathlib.PurePath):
                    file = open(v, "rb")
                    files.append((name, (v.name, file, mimetypes.guess_type(v)[0])))
                    self._opened_files.append(file)
                elif isinstance(v, str):
                    data.setdefault(name, []).append(v)
                else:
                    assert isinstance(v, t.BinaryIO)
                    filename = pathlib.Path(getattr(v, "name", "upload-file")).name
                    content_type = mimetypes.guess_type(filename)[0]
                    files.append((name, (filename, v, content_type)))
        headers.pop("content-type", None)
        return self.client.build_request(
            "POST", endpoint.route, data=data, files=files, headers=headers
        )

    def _deserialize_output(self, payload: Payload, endpoint: ClientEndpoint) -> t.Any:
        from _bentoml_sdk.io_models import IORootModel

        data = iter(payload.data)
        if (endpoint.output.get("type")) == "string":
            content = bytes(next(data))
            if endpoint.output.get("format") == "binary":
                return content
            return content.decode("utf-8")
        elif endpoint.output_spec is not None:
            model = self.serde.deserialize_model(payload, endpoint.output_spec)
            if isinstance(model, IORootModel):
                return model.root  # type: ignore
            return model
        else:
            return self.serde.deserialize(payload, endpoint.output)

    def call(self, __name: str, /, *args: t.Any, **kwargs: t.Any) -> t.Any:
        try:
            endpoint = self.endpoints[__name]
        except KeyError:
            raise NotFound(f"Endpoint {__name} not found") from None
        if endpoint.stream_output:
            return self._get_stream(endpoint, args, kwargs)
        else:
            return self._call(endpoint, args, kwargs)

    @abstractmethod
    def _call(
        self,
        endpoint: ClientEndpoint,
        args: t.Sequence[t.Any],
        kwargs: dict[str, t.Any],
        *,
        headers: t.Mapping[str, str] | None = None,
    ) -> t.Any: ...

    @abstractmethod
    def _get_stream(
        self, endpoint: ClientEndpoint, args: t.Any, kwargs: t.Any
    ) -> t.Any: ...


class SyncHTTPClient(HTTPClient[httpx.Client]):
    """A synchronous client for BentoML service.

    .. note:: Inner usage ONLY
    """

    client_cls = httpx.Client

    def __enter__(self: T) -> T:
        return self

    def __exit__(self, exc_type: t.Any, exc: t.Any, tb: t.Any) -> None:
        return self.close()

    def is_ready(self, timeout: int | None = None) -> bool:
        try:
            resp = self.client.get(
                "/readyz", timeout=timeout or httpx.USE_CLIENT_DEFAULT
            )
            return resp.status_code == 200
        except httpx.TimeoutException:
            logger.warn("Timed out waiting for runner to be ready")
            return False

    def close(self) -> None:
        if "client" in vars(self):
            self.client.close()

    def _get_stream(
        self, endpoint: ClientEndpoint, args: t.Any, kwargs: t.Any
    ) -> t.Generator[t.Any, None, None]:
        resp = self._call(endpoint, args, kwargs)
        for data in resp:
            yield data

    def request(self, method: str, url: str, **kwargs: t.Any) -> httpx.Response:
        return self.client.request(method, url, **kwargs)

    def _submit(
        self, __endpoint: ClientEndpoint, /, *args: t.Any, **kwargs: t.Any
    ) -> Task:
        try:
            req = self._build_request(__endpoint, args, kwargs, {})
            req.url = req.url.copy_with(path=f"{__endpoint.route}/submit")
            resp = self.client.send(req)
            if resp.is_error:
                resp.read()
                raise BentoMLException(
                    f"Error making request: {resp.status_code}: {resp.text}",
                    error_code=HTTPStatus(resp.status_code),
                )
            data = resp.json()
            return Task(data["task_id"], __endpoint, self)
        finally:
            for f in self._opened_files:
                f.close()
            self._opened_files.clear()

    def _call(
        self,
        endpoint: ClientEndpoint,
        args: t.Sequence[t.Any],
        kwargs: dict[str, t.Any],
        *,
        headers: t.Mapping[str, str] | None = None,
    ) -> t.Any:
        try:
            req = self._build_request(endpoint, args, kwargs, headers or {})
            resp = self.client.send(req, stream=endpoint.stream_output)
            if resp.is_error:
                resp.read()
                raise map_exception(resp)
            if endpoint.stream_output:
                return self._parse_stream_response(endpoint, resp)
            elif endpoint.output.get("type") == "file":
                # file responses are always raw binaries whatever the serde is
                return self._parse_file_response(endpoint, resp)
            else:
                return self._parse_response(endpoint, resp)
        finally:
            for f in self._opened_files:
                f.close()
            self._opened_files.clear()

    def _parse_response(self, endpoint: ClientEndpoint, resp: httpx.Response) -> t.Any:
        payload = Payload((resp.read(),), resp.headers)
        return self._deserialize_output(payload, endpoint)

    def _parse_stream_response(
        self, endpoint: ClientEndpoint, resp: httpx.Response
    ) -> t.Generator[t.Any, None, None]:
        try:
            for data in resp.iter_bytes():
                yield self._deserialize_output(Payload((data,), resp.headers), endpoint)
        finally:
            resp.close()

    def _parse_file_response(
        self, endpoint: ClientEndpoint, resp: httpx.Response
    ) -> pathlib.Path | Image.Image:
        from multipart.multipart import parse_options_header
        from PIL import Image

        content_disposition = resp.headers.get("content-disposition")
        content_type = resp.headers.get("content-type", "")
        filename: str | None = None
        if endpoint.output.get("pil"):
            image_formats = (
                [content_type[6:]] if content_type.startswith("image/") else None
            )
            return Image.open(io.BytesIO(resp.read()), formats=image_formats)
        if content_disposition:
            _, options = parse_options_header(content_disposition)
            if b"filename" in options:
                filename = str(
                    options[b"filename"],
                    resp.charset_encoding or "utf-8",
                    errors="ignore",
                )

        with tempfile.NamedTemporaryFile(
            "wb", suffix=filename, dir=self._temp_dir.name, delete=False
        ) as f:
            f.write(resp.read())
        return pathlib.Path(f.name)


class AsyncHTTPClient(HTTPClient[httpx.AsyncClient]):
    """An asynchronous client for BentoML service.

    .. note:: Inner usage ONLY
    """

    client_cls = httpx.AsyncClient

    async def is_ready(self, timeout: int | None = None) -> bool:
        try:
            resp = await self.client.get(
                "/readyz", timeout=timeout or httpx.USE_CLIENT_DEFAULT
            )
            return resp.status_code == 200
        except httpx.TimeoutException:
            logger.warn("Timed out waiting for runner to be ready")
            return False

    async def _get_stream(
        self, endpoint: ClientEndpoint, args: t.Any, kwargs: t.Any
    ) -> t.AsyncGenerator[t.Any, None]:
        resp = await self._call(endpoint, args, kwargs)
        assert inspect.isasyncgen(resp)
        async for data in resp:
            yield data

    async def __aenter__(self: T) -> T:
        return self

    async def __aexit__(self, *args: t.Any) -> None:
        return await self.close()

    async def request(self, method: str, url: str, **kwargs: t.Any) -> httpx.Response:
        return await self.client.request(method, url, **kwargs)

    async def _submit(
        self, __endpoint: ClientEndpoint, /, *args: t.Any, **kwargs: t.Any
    ) -> AsyncTask:
        try:
            req = self._build_request(__endpoint, args, kwargs, {})
            req.url = req.url.copy_with(path=f"{__endpoint.route}/submit")
            resp = await self.client.send(req)
            if resp.is_error:
                resp.read()
                raise BentoMLException(
                    f"Error making request: {resp.status_code}: {resp.text}",
                    error_code=HTTPStatus(resp.status_code),
                )
            data = resp.json()
            return AsyncTask(data["task_id"], __endpoint, self)
        finally:
            for f in self._opened_files:
                f.close()
            self._opened_files.clear()

    async def _call(
        self,
        endpoint: ClientEndpoint,
        args: t.Sequence[t.Any],
        kwargs: dict[str, t.Any],
        *,
        headers: t.Mapping[str, str] | None = None,
    ) -> t.Any:
        try:
            req = self._build_request(endpoint, args, kwargs, headers or {})
            resp = await self.client.send(req, stream=endpoint.stream_output)
            if resp.is_error:
                await resp.aread()
                raise map_exception(resp)
            if endpoint.stream_output:
                return self._parse_stream_response(endpoint, resp)
            elif endpoint.output.get("type") == "file":
                # file responses are always raw binaries whatever the serde is
                return await self._parse_file_response(endpoint, resp)
            else:
                return await self._parse_response(endpoint, resp)
        finally:
            for f in self._opened_files:
                f.close()
            self._opened_files.clear()

    async def _parse_response(
        self, endpoint: ClientEndpoint, resp: httpx.Response
    ) -> t.Any:
        data = await resp.aread()
        return self._deserialize_output(Payload((data,), resp.headers), endpoint)

    async def _parse_stream_response(
        self, endpoint: ClientEndpoint, resp: httpx.Response
    ) -> t.AsyncGenerator[t.Any, None]:
        try:
            async for data in resp.aiter_bytes():
                yield self._deserialize_output(Payload((data,), resp.headers), endpoint)
        finally:
            await resp.aclose()

    async def _parse_file_response(
        self, endpoint: ClientEndpoint, resp: httpx.Response
    ) -> pathlib.Path | Image.Image:
        from multipart.multipart import parse_options_header
        from PIL import Image

        content_disposition = resp.headers.get("content-disposition")
        content_type = resp.headers.get("content-type", "")
        filename: str | None = None
        if endpoint.output.get("pil"):
            image_formats = (
                [content_type[6:]] if content_type.startswith("image/") else None
            )
            return Image.open(io.BytesIO(await resp.aread()), formats=image_formats)
        if content_disposition:
            _, options = parse_options_header(content_disposition)
            if b"filename" in options:
                filename = str(
                    options[b"filename"],
                    resp.charset_encoding or "utf-8",
                    errors="ignore",
                )

        with tempfile.NamedTemporaryFile(
            "wb", suffix=filename, dir=self._temp_dir.name, delete=False
        ) as f:
            f.write(await resp.aread())
        return pathlib.Path(f.name)

    async def close(self) -> None:
        if "client" in vars(self):
            await self.client.aclose()
