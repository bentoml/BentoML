from __future__ import annotations

import typing as t
import logging
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

from ....exceptions import StateException

if TYPE_CHECKING:
    from ..runner import Runner


logger = logging.getLogger(__name__)


class RunnerHandle(ABC):
    @abstractmethod
    def __init__(self, runner: Runner) -> None:
        ...

    @abstractmethod
    def run_method(
        self,
        method_name: str,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> t.Any:
        ...

    @abstractmethod
    async def async_run_method(
        self,
        method_name: str,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> t.Any:
        ...


class DummyRunnerHandle(RunnerHandle):
    def __init__(  # pylint: disable=super-init-not-called
        self, runner: Runner | None = None
    ) -> None:
        pass

    def run_method(
        self,
        method_name: str,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> t.Any:
        raise StateException("Runner is not initialized")

    async def async_run_method(
        self,
        method_name: str,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> t.Any:
        raise StateException("Runner is not initialized")
