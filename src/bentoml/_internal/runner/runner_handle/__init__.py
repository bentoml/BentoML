from __future__ import annotations

import typing as t
import logging
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

from ....exceptions import StateException

if TYPE_CHECKING:
    from ..runner import Runner
    from ..runner import RunnerMethod
    from ..runner import AbstractRunner

    R = t.TypeVar("R")
    P = t.ParamSpec("P")


logger = logging.getLogger(__name__)


class RunnerHandle(ABC):
    @abstractmethod
    def __init__(self, runner: AbstractRunner) -> None:
        ...

    @abstractmethod
    async def is_ready(self, timeout: int) -> bool:
        return True

    @abstractmethod
    def run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        ...

    @abstractmethod
    async def async_run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        ...


class DummyRunnerHandle(RunnerHandle):
    def __init__(  # pylint: disable=super-init-not-called
        self, runner: Runner | None = None
    ) -> None:
        pass

    async def is_ready(self, timeout: int) -> bool:
        return True

    def run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, t.Any, t.Any],
        *args: t.Any,
        **kwargs: t.Any,
    ) -> t.Any:
        raise StateException(
            f"Runner is not initialized. Make sure to include '{self!r}' to your service definition."
        )

    async def async_run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, t.Any, t.Any],
        *args: t.Any,
        **kwargs: t.Any,
    ) -> t.Any:
        raise StateException(
            f"Runner is not initialized. Make sure to include '{self!r}' to your service definition."
        )
