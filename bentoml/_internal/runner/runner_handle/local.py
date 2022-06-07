from __future__ import annotations

import typing as t
import functools
from typing import TYPE_CHECKING

from bentoml._internal.runner.utils import Params
from bentoml._internal.runner.container import Payload
from bentoml._internal.runner.container import AutoContainer

from . import RunnerHandle

if TYPE_CHECKING:
    from ..runner import Runner
    from ..runner import RunnerMethod

    P = t.ParamSpec("P")
    R = t.TypeVar("R")


class LocalRunnerRef(RunnerHandle):
    def __init__(self, runner: Runner) -> None:  # pylint: disable=super-init-not-called
        self._runnable = runner.runnable_class()

    def run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        if __bentoml_method.config.batchable:
            inp_batch_dim = __bentoml_method.config.batch_dim[0]

            payload_params = Params[Payload](*args, **kwargs).map(
                lambda arg: AutoContainer.to_payload(arg, batch_dim=inp_batch_dim)
            )

            if not payload_params.map(lambda i: i.batch_size).all_equal():
                raise ValueError(
                    "All batchable arguments must have the same batch size."
                )

        return getattr(self._runnable, __bentoml_method.name)(*args, **kwargs)

    async def async_run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        import anyio

        method = getattr(self._runnable, __bentoml_method.name)
        return await anyio.to_thread.run_sync(
            functools.partial(method, **kwargs), *args, limiter=anyio.CapacityLimiter(1)
        )
