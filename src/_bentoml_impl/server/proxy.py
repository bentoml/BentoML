from __future__ import annotations

import asyncio
import contextlib
import logging
import typing as t

import aiohttp
import anyio
import httpx
from pyparsing import cast
from starlette.requests import Request

from _bentoml_sdk import Service
from bentoml import get_current_service
from bentoml import server_context
from bentoml._internal.utils import expand_envs
from bentoml.exceptions import BentoMLConfigException

if t.TYPE_CHECKING:
    from anyio.abc._subprocesses import Process
    from starlette.applications import Starlette

logger = logging.getLogger("bentoml.server")


async def _check_health(client: aiohttp.ClientSession, health_endpoint: str) -> bool:
    try:
        response = await client.get(health_endpoint, timeout=aiohttp.ClientTimeout(5))
        if response.status == 404:
            raise BentoMLConfigException(
                f"Health endpoint {health_endpoint} not found (404). Please make sure the health "
                "endpoint is correctly configured in the service config."
            )
        return response.ok
    except aiohttp.ClientError:
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
        # Only start process for the first worker or when replicate_process is True
        should_start_process = (
            service.config.get("replicate_process", False)
            or server_context.worker_index == 1
        )
        async with contextlib.AsyncExitStack() as stack:
            proxy_port = service.config.get("http", {}).get("proxy_port", 8000)
            proxy_url = f"http://localhost:{proxy_port}"
            proc: Process | None = None
            if (
                instance_client := getattr(server_instance, "client", None)
            ) is not None and isinstance(instance_client, aiohttp.ClientSession):
                # TODO: support aiohttp client
                client = instance_client
            else:
                client = await stack.enter_async_context(
                    aiohttp.ClientSession(
                        base_url=proxy_url,
                        timeout=aiohttp.ClientTimeout(),
                        connector=aiohttp.TCPConnector(limit=512),
                    )
                )
            if should_start_process:
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
                proc = await anyio.open_process(cmd, stdout=None, stderr=None)
            while proc is None or proc.returncode is None:
                if await _check_health(client, health_endpoint):
                    break
                await anyio.sleep(0.5)
            else:
                raise RuntimeError(
                    "Service process exited before becoming healthy, see the error above"
                )

            app.state.client = client
            try:
                state = {
                    "proc": proc,
                    "client": await stack.enter_async_context(
                        httpx.AsyncClient(base_url=proxy_url, timeout=None)
                    ),  # For backward compatibility
                }
                service.context.state.update(state)
                yield state
            finally:
                if proc is not None:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=30.0)
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()

    assert service.has_custom_command(), "Service does not have custom command"
    app = fastapi.FastAPI(lifespan=lifespan)

    # TODO: support websocket endpoints
    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def reverse_proxy(request: Request, path: str):
        from contextlib import AsyncExitStack

        import yarl
        from starlette.background import BackgroundTask

        url = yarl.URL.build(path=f"/{path}", query=request.url.query or None)
        client = t.cast(aiohttp.ClientSession, app.state.client)
        headers = dict(request.headers)
        headers.pop("host", None)
        stack = AsyncExitStack()
        try:
            resp = await stack.enter_async_context(
                client.request(
                    url=url,
                    method=request.method,
                    headers=headers,
                    data=request.stream(),
                )
            )
        except aiohttp.ClientError as e:
            await stack.aclose()
            return fastapi.Response(content=str(e), status_code=503)
        except Exception:
            logger.exception("Error occurred while proxying request")
            await stack.aclose()
            return fastapi.Response(status_code=500)

        return StreamingResponse(
            resp.content.iter_any(),
            status_code=resp.status,
            headers=resp.headers,
            background=BackgroundTask(stack.aclose),
        )

    return app
