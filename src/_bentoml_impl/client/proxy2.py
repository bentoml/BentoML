from __future__ import annotations

import asyncio
import functools
import inspect
import io
import json
import logging
import os
import pathlib
import tempfile
import time
import typing as t
from functools import cached_property
from http import HTTPStatus
from urllib.parse import urljoin
from urllib.parse import urlparse

import aiohttp
import anyio.from_thread
import attr
from multidict import CIMultiDict

from _bentoml_sdk.service.config import ServiceConfig
from bentoml import __version__
from bentoml._internal.io_descriptors.base import IODescriptor
from bentoml._internal.utils.uri import uri_to_path
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import NotFound
from bentoml.exceptions import ServiceUnavailable

from ..serde import Payload
from .base import AbstractClient
from .base import ClientEndpoint
from .base import ClientFileManager
from .task import AsyncTask
from .task import ResultStatus
from .task import Task

T = t.TypeVar("T")
if t.TYPE_CHECKING:
    from PIL import Image

    P = t.ParamSpec("P")

    from _bentoml_sdk import Service
    from bentoml._internal.external_typing import ASGIApp

    from ..serde import Serde


logger = logging.getLogger("bentoml.io")


async def map_exception(resp: aiohttp.ClientResponse) -> BentoMLException:
    status = HTTPStatus(resp.status)
    exc = BentoMLException.error_mapping.get(status, BentoMLException)
    return exc(await resp.text(), error_code=status)


class SessionManager:
    def __init__(
        self,
        url: str,
        timeout: float,
        headers: dict[str, str],
        app: ASGIApp | None = None,
        max_age: float = 300.0,
        max_requests: int = 100,
    ) -> None:
        self.max_age = max_age
        self.max_requests = max_requests

        self._session: aiohttp.ClientSession | None = None
        self._created_at = 0.0
        self._request_count = 0

        parsed = urlparse(url)
        connector: aiohttp.BaseConnector | None = None
        if app is not None:
            from aiohttp_asgi_connector import ASGIApplicationConnector

            connector = ASGIApplicationConnector(app)  # type: ignore[arg-type]
            url = "http://127.0.0.1:3000"
        elif parsed.scheme == "file":
            from aiohttp import UnixConnector

            connector = UnixConnector(path=uri_to_path(url))
            url = "http://127.0.0.1:3000"
        elif parsed.scheme == "tcp":
            url = f"http://{parsed.netloc}"
        self._make_client = lambda: aiohttp.ClientSession(
            base_url=url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
            connector=connector,
        )

    def _should_refresh(self) -> bool:
        if (time.monotonic() - self._created_at) > self.max_age:
            return True
        if self._request_count > self.max_requests:
            return True
        return False

    async def get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._should_refresh():
            if self._session is not None:
                await self._session.close()
            self._session = self._make_client()
            self._created_at = time.monotonic()
            self._request_count = 0
        self._request_count += 1
        return self._session

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


@attr.define(slots=False)
class AsyncClient(AbstractClient):
    url: str
    endpoints: dict[str, ClientEndpoint] = attr.field(factory=dict)
    media_type: str = "application/json"
    timeout: float = 30
    default_headers: dict[str, str] = attr.field(factory=dict)
    app: ASGIApp | None = None
    server_ready_timeout: float | None = None
    service: Service[t.Any] | None = None

    _file_manager: ClientFileManager = attr.field(init=False, factory=ClientFileManager)
    _temp_dir: tempfile.TemporaryDirectory[str] = attr.field(init=False)
    _setup_done: bool = attr.field(init=False, default=False)

    def __init__(
        self,
        url: str,
        *,
        media_type: str = "application/json",
        service: Service[t.Any] | None = None,
        server_ready_timeout: float | None = None,
        token: str | None = None,
        timeout: float = 30,
        max_age: float = 300.0,
        max_requests: int = 100,
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

        self._readyz_endpoint = "/readyz"

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
            self._readyz_endpoint = service.config.get("endpoints", {}).get(
                "readyz", self._readyz_endpoint
            )
        self.__attrs_init__(  # type: ignore
            url=url,
            endpoints=routes,
            media_type=media_type,
            default_headers=default_headers,
            timeout=timeout,
            app=app,
            server_ready_timeout=server_ready_timeout,
            service=service,
        )
        self._session_manager = SessionManager(
            url=url,
            timeout=timeout,
            headers=default_headers,
            app=app,
            max_age=max_age,
            max_requests=max_requests,
        )

    @_temp_dir.default  # type: ignore
    def default_temp_dir(self) -> tempfile.TemporaryDirectory[str]:
        return tempfile.TemporaryDirectory(prefix="bentoml-client-")

    @cached_property
    def serde(self) -> Serde:
        from ..serde import ALL_SERDE

        return ALL_SERDE[self.media_type]()

    async def close(self) -> None:
        await self._session_manager.close()
        self._file_manager.close()
        self._temp_dir.cleanup()

    def call(self, __name: str, /, *args: t.Any, **kwargs: t.Any) -> t.Any:
        try:
            endpoint = self.endpoints[__name]
        except KeyError:
            raise NotFound(f"Endpoint {__name} not found") from None
        if endpoint.stream_output:
            return self._get_stream(endpoint, args, kwargs)
        else:
            return self._call(endpoint, args, kwargs)

    async def _setup(self) -> None:
        if self._setup_done:
            return

        if self.app is None and (
            self.server_ready_timeout is None or self.server_ready_timeout > 0
        ):
            await self.wait_until_server_ready(self.server_ready_timeout)
        if self.service is None:
            schema_url = urljoin(self.url, "/schema.json")
            client = await self._session_manager.get_session()
            async with client.get("/schema.json") as resp:
                if not resp.ok:
                    raise BentoMLException(f"Failed to fetch schema from {schema_url}")
                for route in (await resp.json())["routes"]:
                    self.endpoints[route["name"]] = ClientEndpoint(
                        name=route["name"],
                        route=route["route"],
                        input=route["input"],
                        output=route["output"],
                        doc=route.get("doc"),
                        stream_output=route["output"].get("is_stream", False),
                        is_task=route.get("is_task", False),
                    )
        self._setup_endpoints()

    async def wait_until_server_ready(self, timeout: float | None = None) -> None:
        if timeout is None:
            timeout = self.timeout
        end = time.monotonic() + timeout
        while (current := time.monotonic()) < end:
            client = await self._session_manager.get_session()
            try:
                async with client.get(
                    self._readyz_endpoint, timeout=aiohttp.ClientTimeout(end - current)
                ) as resp:
                    if resp.status == 200:
                        return
            except (aiohttp.ClientError, aiohttp.ServerTimeoutError):
                pass
        raise ServiceUnavailable(f"Server is not ready after {timeout} seconds")

    async def is_ready(self, timeout: int | None = None) -> bool:
        client = await self._session_manager.get_session()
        try:
            resp = await client.get(
                self._readyz_endpoint,
                timeout=aiohttp.ClientTimeout(timeout) if timeout else None,
            )
            return resp.status == 200
        except aiohttp.ServerTimeoutError:
            logger.warning("Timed out waiting for runner to be ready")
            return False

    async def __aenter__(self) -> t.Self:
        await self._setup()
        return self

    async def __aexit__(self, *args: t.Any) -> None:
        return await self.close()

    async def _submit(
        self, __endpoint: ClientEndpoint, /, *args: t.Any, **kwargs: t.Any
    ) -> AsyncTask:
        client = await self._session_manager.get_session()
        try:
            headers = CIMultiDict[str]()
            body = self._build_payload(__endpoint, args, kwargs, headers)
            url = f"{__endpoint.route}/submit"
            async with client.post(url, data=body, headers=headers) as resp:
                if not resp.ok:
                    raise BentoMLException(
                        f"Error making request: {resp.status}: {await resp.text()}",
                        error_code=HTTPStatus(resp.status),
                    )
            data = await resp.json()
            return AsyncTask(data["task_id"], __endpoint, self)
        finally:
            self._file_manager.close()

    async def _get_stream(
        self, endpoint: ClientEndpoint, args: t.Any, kwargs: t.Any
    ) -> t.AsyncGenerator[t.Any, None]:
        resp = await self._call(endpoint, args, kwargs)
        assert inspect.isasyncgen(resp)
        async for data in resp:
            yield data

    async def _call(
        self,
        endpoint: ClientEndpoint,
        args: t.Sequence[t.Any],
        kwargs: dict[str, t.Any],
        *,
        headers: t.Mapping[str, str] | None = None,
    ) -> t.Any:
        client = await self._session_manager.get_session()
        try:
            headers = CIMultiDict({"Content-Type": self.media_type, **(headers or {})})
            body = self._build_payload(endpoint, args, kwargs, headers)
            resp = await client.post(endpoint.route, data=body, headers=headers)
            if not resp.ok:
                raise await map_exception(resp)
            if endpoint.stream_output:
                return self._parse_stream_response(endpoint, resp)
            elif endpoint.output.get("type") == "file":
                # file responses are always raw binaries whatever the serde is
                return await self._parse_file_response(endpoint, resp)
            else:
                return await self._parse_response(endpoint, resp)
        finally:
            self._file_manager.close()

    # ==========================
    # Request builders
    # ==========================

    def _build_payload(
        self,
        endpoint: ClientEndpoint,
        args: t.Sequence[t.Any],
        kwargs: dict[str, t.Any],
        headers: dict[str, str],
    ) -> t.Any:
        from opentelemetry import propagate

        from _bentoml_sdk.io_models import IORootModel

        propagate.inject(headers)
        if endpoint.input_spec is not None:
            model = endpoint.input_spec.from_inputs(*args, **kwargs)
            if (
                not isinstance(model, IORootModel)
                and model.multipart_fields
                and self.media_type == "application/json"
            ):
                return self._build_multipart(endpoint, model, headers)
            elif isinstance(rendered := model.model_dump(), (str, bytes)):
                headers.update({"content-type": model.mime_type()})
                return rendered
            else:
                payload = self.serde.serialize_model(model)
                headers.update(payload.headers)
                return payload.aiter_bytes()

        assert self.media_type == "application/json", (
            "Non-JSON request is not supported"
        )
        if endpoint.input.get("root_input", False):
            if len(args) > 1 or kwargs:
                raise TypeError("Expected one positional argument for root input")
            if not args:
                return None
            value = args[0]
            passthrough = False
            content = None
            if "properties" in endpoint.input:
                kwargs = value
                args = ()
                passthrough = True
            elif endpoint.input.get("type") == "file":
                file = self._file_manager.get_file(value)
                if isinstance(file, str):
                    content = file
                else:
                    file_io, content_type = file[1:]
                    content = iter(lambda: file_io.read(4096), b"")
                    if content_type:
                        headers.update({"content-type": content_type})
            elif isinstance(value, (str, bytes)):
                content = value.encode("utf-8") if isinstance(value, str) else value
                headers.update({"content-type": "text/plain"})
            else:
                payload = self.serde.serialize(value, endpoint.input)
                headers.update(payload.headers)
                content = payload.aiter_bytes()
            if not passthrough:
                return content

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
        if has_file:
            return self._build_multipart(endpoint, kwargs, headers)
        payload = self.serde.serialize(kwargs, endpoint.input)
        headers.update(payload.headers)
        return payload.aiter_bytes()

    def _build_multipart(
        self,
        endpoint: ClientEndpoint,
        model: IODescriptor | dict[str, t.Any],
        headers: dict[str, str],
    ) -> aiohttp.FormData:
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
            fields = {t.cast(str, k): getattr(model, k) for k in model.model_fields}
        data: dict[str, t.Any] = {}
        files: list[tuple[str, tuple[str, t.IO[bytes], str | None]]] = []

        for name, value in fields.items():
            if not is_file_field(name):
                data[name] = json.dumps(value)
                continue
            if not isinstance(value, (list, tuple)):
                value = [value]

            for v in value:
                file = self._file_manager.get_file(v)
                if isinstance(file, str):
                    data[name] = file
                else:
                    files.append((name, file))
        headers.pop("content-type", None)
        payload = aiohttp.FormData()
        for key, val in data.items():
            payload.add_field(key, val)
        for key, (filename, fileobj, content_type) in files:
            payload.add_field(
                key,
                fileobj,
                filename=filename,
                content_type=content_type,
            )
        return payload

    # ==========================
    # Response parsers
    # ==========================

    def _deserialize_output(self, payload: Payload, endpoint: ClientEndpoint) -> t.Any:
        from _bentoml_sdk.io_models import IORootModel

        data = payload.iter_bytes()
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

    async def _parse_response(
        self, endpoint: ClientEndpoint, resp: aiohttp.ClientResponse
    ) -> t.Any:
        data = await resp.read()
        return self._deserialize_output(Payload((data,), resp.headers), endpoint)

    async def _parse_stream_response(
        self, endpoint: ClientEndpoint, resp: aiohttp.ClientResponse
    ) -> t.AsyncGenerator[t.Any, None]:
        async for data in resp.content.iter_chunked(1024 * 8):
            yield self._deserialize_output(Payload((data,), resp.headers), endpoint)

    async def _parse_file_response(
        self, endpoint: ClientEndpoint, resp: aiohttp.ClientResponse
    ) -> pathlib.Path | Image.Image:
        from PIL import Image
        from python_multipart.multipart import parse_options_header

        content_disposition = resp.headers.get("content-disposition")
        content_type = resp.headers.get("content-type", "")
        filename: str | None = None
        if endpoint.output.get("pil"):
            image_formats = (
                [content_type[6:]] if content_type.startswith("image/") else None
            )
            return Image.open(io.BytesIO(await resp.read()), formats=image_formats)
        if content_disposition:
            _, options = parse_options_header(content_disposition)
            if b"filename" in options:
                filename = str(options[b"filename"], "utf8", errors="ignore")

        with tempfile.NamedTemporaryFile(
            "wb", suffix=filename, dir=self._temp_dir.name, delete=False
        ) as f:
            f.write(await resp.read())
        return pathlib.Path(f.name)

    async def _get_task_status(
        self, __endpoint: ClientEndpoint, /, task_id: str
    ) -> ResultStatus:
        client = await self._session_manager.get_session()
        async with client.get(
            f"{__endpoint.route}/status", params={"task_id": task_id}
        ) as resp:
            if not resp.ok:
                raise await map_exception(resp)
            data = await resp.json()
            return ResultStatus(data["status"])

    async def _cancel_task(self, __endpoint: ClientEndpoint, /, task_id: str) -> None:
        client = await self._session_manager.get_session()
        async with client.put(
            f"{__endpoint.route}/cancel", params={"task_id": task_id}
        ) as resp:
            if not resp.ok:
                raise await map_exception(resp)

    async def _retry_task(
        self, __endpoint: ClientEndpoint, /, task_id: str
    ) -> AsyncTask:
        client = await self._session_manager.get_session()
        async with client.post(
            f"{__endpoint.route}/retry", params={"task_id": task_id}
        ) as resp:
            if not resp.ok:
                raise await map_exception(resp)
            data = await resp.json()
            return AsyncTask(data["task_id"], __endpoint, self)

    async def _get_task_result(
        self, __endpoint: ClientEndpoint, /, task_id: str
    ) -> t.Any:
        client = await self._session_manager.get_session()
        async with client.get(
            f"{__endpoint.route}/get", params={"task_id": task_id}
        ) as resp:
            if not resp.ok:
                raise await map_exception(resp)
            if (
                __endpoint.output.get("type") == "file"
                and self.media_type == "application/json"
            ):
                return await self._parse_file_response(__endpoint, resp)
            else:
                return await self._parse_response(__endpoint, resp)


class SyncClient(AbstractClient):
    def __init__(self, async_client: AsyncClient) -> None:
        self._async_client = async_client

    @staticmethod
    def run_async(
        callable: t.Callable[P, t.Awaitable[T]], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        return anyio.from_thread.run(functools.partial(callable, *args, **kwargs))

    def call(self, __name: str, /, *args: t.Any, **kwargs: t.Any) -> t.Any:
        coro = self._async_client.call(__name, *args, **kwargs)
        if inspect.isasyncgen(coro):

            def generator() -> t.Generator[t.Any, None, None]:
                while True:
                    try:
                        yield anyio.from_thread.run(coro.__anext__)
                    except StopAsyncIteration:
                        break

            return generator()

        return self.run_async(lambda: coro)

    def _get_task_status(self, __endpoint: ClientEndpoint, /, task_id: str) -> t.Any:
        return self.run_async(self._async_client._get_task_status, __endpoint, task_id)

    def _cancel_task(self, __endpoint: ClientEndpoint, /, task_id: str) -> None:
        return self.run_async(self._async_client._cancel_task, __endpoint, task_id)

    def _retry_task(self, __endpoint: ClientEndpoint, /, task_id: str) -> t.Any:
        return self.run_async(self._async_client._retry_task, __endpoint, task_id)

    def _get_task_result(self, __endpoint: ClientEndpoint, /, task_id: str) -> t.Any:
        return self.run_async(self._async_client._get_task_result, __endpoint, task_id)

    def _submit(
        self, __endpoint: ClientEndpoint, /, *args: t.Any, **kwargs: t.Any
    ) -> Task:
        async_task = self.run_async(
            self._async_client._submit, __endpoint, *args, **kwargs
        )
        return Task(async_task.id, __endpoint, self)


class RemoteProxy(AbstractClient, t.Generic[T]):
    """A remote proxy of the passed in service that has the same interfaces"""

    def __init__(
        self,
        url: str,
        *,
        service: Service[T] | None = None,
        media_type: str = "application/vnd.bentoml+pickle",
        app: ASGIApp | None = None,
    ) -> None:
        from bentoml.container import BentoMLContainer

        if service is not None:
            svc_config: dict[str, ServiceConfig] = (
                BentoMLContainer.config.services.get()
            )
            timeout = (
                svc_config.get(service.name, {}).get("traffic", {}).get("timeout") or 60
            ) * 1.01  # get the service timeout add 1% margin for the client
            runner_conn_config = svc_config.get(service.name, {}).get(
                "runner_connection", {}
            )
            max_age = runner_conn_config.get("max_age", 300.0)
            max_requests = runner_conn_config.get("max_requests", 100)
        else:
            timeout = 60
            max_age = 300.0
            max_requests = 100
        self._async = AsyncClient(
            url,
            media_type=media_type,
            service=service,
            timeout=timeout,
            server_ready_timeout=0,
            app=app,
            max_age=max_age,
            max_requests=max_requests,
        )
        self._sync = SyncClient(self._async)
        if service is not None:
            self._inner = service.inner
        else:
            self._inner = None

    async def __aenter__(self) -> t.Self:
        await self._async.__aenter__()
        self.endpoints = self._sync.endpoints = self._async.endpoints
        self._sync._setup_endpoints()
        self._setup_endpoints()
        return self

    @property
    def to_async(self) -> AsyncClient:
        return self._async

    @property
    def to_sync(self) -> SyncClient:
        return self._sync

    @property
    def client_url(self) -> str:
        return str(self._async.client._base_url)  # type: ignore[attr-defined]

    async def is_ready(self, timeout: int | None = None) -> bool:
        return await self._async.is_ready(timeout=timeout)

    async def close(self) -> None:
        await self._async.close()

    def as_service(self) -> T:
        return t.cast(T, self)

    def call(self, __name: str, /, *args: t.Any, **kwargs: t.Any) -> t.Any:
        if self._inner is None:
            raise BentoMLException(
                "The proxy is not callable when the service is not provided. Please use `.to_async` or `.to_sync` property."
            )
        original_func = getattr(self._inner, __name)
        if not hasattr(original_func, "func"):
            raise BentoMLException(f"calling non-api method {__name} is not allowed")
        original_func = original_func.func
        while isinstance(original_func, functools.partial):
            original_func = original_func.func
        is_async_func = (
            asyncio.iscoroutinefunction(original_func)
            or (
                callable(original_func)
                and asyncio.iscoroutinefunction(original_func.__call__)  # type: ignore
            )
            or inspect.isasyncgenfunction(original_func)
        )
        if is_async_func:
            return self._async.call(__name, *args, **kwargs)
        else:
            return self._sync.call(__name, *args, **kwargs)

    def _submit(
        self, __endpoint: ClientEndpoint, /, *args: t.Any, **kwargs: t.Any
    ) -> t.Any:
        original_func = getattr(self._inner, __endpoint.name)
        if not hasattr(original_func, "func"):
            raise BentoMLException(
                f"calling non-api method {__endpoint.name} is not allowed"
            )
        original_func = original_func.func
        while isinstance(original_func, functools.partial):
            original_func = original_func.func
        is_async_func = (
            asyncio.iscoroutinefunction(original_func)
            or (
                callable(original_func)
                and asyncio.iscoroutinefunction(original_func.__call__)  # type: ignore
            )
            or inspect.isasyncgenfunction(original_func)
        )
        if is_async_func:
            return self._async._submit(__endpoint, *args, **kwargs)
        else:
            return self._sync._submit(__endpoint, *args, **kwargs)
