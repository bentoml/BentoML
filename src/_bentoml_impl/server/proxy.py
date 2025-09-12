from __future__ import annotations

import contextlib
import logging
import typing as t

import anyio
import httpx
from pyparsing import cast
from starlette.requests import Request

from _bentoml_sdk import Service
from bentoml import get_current_service
from bentoml._internal.utils import expand_envs
from bentoml.exceptions import BentoMLConfigException

if t.TYPE_CHECKING:
    from starlette.applications import Starlette

logger = logging.getLogger("bentoml.server")


async def _check_health(client: httpx.AsyncClient, health_endpoint: str) -> bool:
    try:
        response = await client.get(health_endpoint, timeout=5.0)
        if response.status_code == 404:
            raise BentoMLConfigException(
                f"Health endpoint {health_endpoint} not found (404). Please make sure the health "
                "endpoint is correctly configured in the service config."
            )
        return response.is_success
    except (httpx.HTTPError, httpx.RequestError):
        return False


def create_proxy_app(service: Service[t.Any]) -> Starlette:
    """A reverse-proxy that forwards all requests to the HTTP server started
    by the custom command.
    """
    import fastapi
    from fastapi.responses import StreamingResponse

    health_endpoint = service.config.get("endpoints", {}).get("livez", "/health")

    @contextlib.asynccontextmanager
    async def lifespan(
        app: fastapi.FastAPI,
    ) -> t.AsyncGenerator[dict[str, t.Any], None]:
        server_instance = get_current_service()
        assert server_instance is not None, "Current service is not initialized"
        async with contextlib.AsyncExitStack() as stack:
            if cmd_getter := getattr(server_instance, "__command__", None):
                if not callable(cmd_getter):
                    raise TypeError(
                        f"__command__ must be a callable that returns a list of strings, got {type(cmd_getter)}"
                    )
                cmd = cast("list[str]", cmd_getter())
            else:
                cmd = service.cmd
            assert cmd is not None, "must have a command"
            cmd = [expand_envs(c) for c in cmd]
            logger.info("Running service with command: %s", " ".join(cmd))
            if (
                instance_client := getattr(server_instance, "client", None)
            ) is not None and isinstance(instance_client, httpx.AsyncClient):
                # TODO: support aiohttp client
                client = instance_client
            else:
                proxy_port = service.config.get("http", {}).get("proxy_port")
                if proxy_port is None:
                    raise BentoMLConfigException(
                        "proxy_port must be set in service config to use custom command"
                    )
                proxy_url = f"http://localhost:{proxy_port}"
                client = await stack.enter_async_context(
                    httpx.AsyncClient(base_url=proxy_url, timeout=None)
                )
            proc = await anyio.open_process(cmd, stdout=None, stderr=None)
            while proc.returncode is None:
                if await _check_health(client, health_endpoint):
                    break
                await anyio.sleep(0.5)
            else:
                raise RuntimeError(
                    "Service process exited before becoming healthy, see the error above"
                )

            app.state.client = client
            try:
                state = {"proc": proc, "client": client}
                service.context.state.update(state)
                yield state
            finally:
                proc.terminate()
                await proc.wait()

    assert service.has_custom_command(), "Service does not have custom command"
    app = fastapi.FastAPI(lifespan=lifespan)

    # TODO: support websocket endpoints
    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def reverse_proxy(request: Request, path: str):
        url = httpx.URL(
            path=f"/{path}", query=request.url.query.encode("utf-8") or None
        )
        client = t.cast(httpx.AsyncClient, app.state.client)
        headers = dict(request.headers)
        headers.pop("host", None)
        req = client.build_request(
            method=request.method, url=url, headers=headers, content=request.stream()
        )
        try:
            resp = await client.send(req, stream=True)
        except httpx.ConnectError:
            return fastapi.Response(503)
        except httpx.RequestError:
            return fastapi.Response(500)

        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers=resp.headers,
            background=resp.aclose,
        )

    return app
