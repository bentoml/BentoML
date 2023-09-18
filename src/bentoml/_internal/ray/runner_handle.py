from __future__ import annotations

import typing as t
import functools

from ..runner import Runner
from ...exceptions import MissingDependencyException
from ..runner.runner import RunnerMethod
from ..runner.runner_handle import RunnerHandle

try:
    import ray
    from ray.serve.deployment import Deployment as RayDeployment
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "'ray' is required in order to use module 'bentoml.ray', install with 'pip install -U \"ray[serve]\"'. For more information, refer to https://docs.ray.io/",
    )

if t.TYPE_CHECKING:
    P = t.ParamSpec("P")
else:
    P = t.TypeVar("P")

R = t.TypeVar("R")


class RayRunnerHandle(RunnerHandle):
    def __init__(self, runner: Runner, ray_deployment: RayDeployment) -> None:
        self._runner = runner
        self._ray_deployment = ray_deployment

    async def is_ready(self, timeout: int) -> bool:
        return True

    def run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R | tuple[R, ...]:
        import anyio

        return t.cast(
            "R | tuple[R, ...]",
            anyio.from_thread.run(
                functools.partial(self.async_run_method, **kwargs),
                __bentoml_method,
                *args,
            ),
        )

    async def async_run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R | tuple[R, ...]:
        return ray.get(
            await getattr(self._ray_deployment, __bentoml_method.name).remote(
                *args, **kwargs
            )
        )
