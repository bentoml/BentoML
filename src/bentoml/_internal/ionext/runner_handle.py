from __future__ import annotations

import typing as t
from http import HTTPStatus

import aiohttp

from ...exceptions import BentoMLException
from ...exceptions import RemoteException
from ...exceptions import ServiceUnavailable
from ..context import component_context
from ..runner.runner_handle import RunnerHandle

if t.TYPE_CHECKING:
    from ..runner import Runner
    from ..runner.runner import RunnerMethod

    P = t.ParamSpec("P")
    R = t.TypeVar("R")


class RemoteRunnerHandle(RunnerHandle):
    def __init__(self, runner: Runner) -> None:
        from .client import AsyncClient

        self.client = AsyncClient.for_runner(
            runner, media_type="application/vnd.bentoml+pickle"
        )
        self.headers = {
            "Bento-Name": component_context.bento_name,
            "Bento-Version": component_context.bento_version,
            "Runner-Name": runner.name,
            "Yatai-Bento-Deployment-Name": component_context.yatai_bento_deployment_name,
            "Yatai-Bento-Deployment-Namespace": component_context.yatai_bento_deployment_namespace,
        }

    async def is_ready(self, timeout: int) -> bool:
        return await self.client.is_ready(timeout, headers=self.headers)

    async def async_run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        try:
            return await self.client._call(
                __bentoml_method.name, args, kwargs, headers=self.headers
            )
        except BentoMLException as e:
            if e.error_code == HTTPStatus.SERVICE_UNAVAILABLE:
                raise ServiceUnavailable(e.message) from None
            raise RemoteException(e.message) from None
        except aiohttp.ClientOSError as e:
            raise RemoteException("Failed to connect to runner server") from e

    async def async_stream_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> t.AsyncGenerator[R, None]:
        result = t.cast(
            t.AsyncGenerator["R", None],
            await self.async_run_method(__bentoml_method, *args, **kwargs),
        )
        async for item in result:
            yield item

    def run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        import functools

        import anyio

        return t.cast(
            "R",
            anyio.from_thread.run(
                functools.partial(self.async_run_method, **kwargs),
                __bentoml_method,
                *args,
            ),
        )
