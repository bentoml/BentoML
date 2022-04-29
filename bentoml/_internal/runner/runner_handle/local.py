from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

from . import RunnerHandle

if TYPE_CHECKING:
    from ..runner import Runner
    from ..runnable import Runnable


class LocalRunnerRef(RunnerHandle):
    _runnable: Runnable

    def __init__(self, runner: Runner) -> None:  # pylint: disable=super-init-not-called
        self._runnable = runner.runnable_class()

    def run_method(self, method_name: str, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return getattr(self._runnable, method_name)(*args, **kwargs)

    async def async_run_method(
        self, method_name: str, *args: t.Any, **kwargs: t.Any
    ) -> t.Any:
        import anyio

        method = getattr(self._runnable, method_name)
        return anyio.to_thread.run_sync(method, *args, **kwargs)
