from __future__ import annotations

import typing as t
import functools
from typing import TYPE_CHECKING

from . import RunnerHandle

if TYPE_CHECKING:
    from ..runner import Runner
    from ..runnable import Runnable


class LocalRunnerRef(RunnerHandle):
    _runnable: Runnable

    def __init__(self, runner: Runner) -> None:  # pylint: disable=super-init-not-called
        self._runnable = runner.runnable_class()

    def run_method(
        self, __bentoml_method_name: str, *args: t.Any, **kwargs: t.Any
    ) -> t.Any:
        args_list = list(args)
        args_list.insert(0, self._runnable)
        args = tuple(args_list)
        return getattr(self._runnable, __bentoml_method_name)(*args, **kwargs)

    async def async_run_method(
        self, __bentoml_method_name: str, *args: t.Any, **kwargs: t.Any
    ) -> t.Any:
        import anyio

        method = getattr(self._runnable, __bentoml_method_name)
        args_list = list(args)
        # args_list.insert(0, self._runnable)
        args = tuple(args_list)
        return await anyio.to_thread.run_sync(
            functools.partial(method, **kwargs), *args, limiter=anyio.CapacityLimiter(1)
        )
