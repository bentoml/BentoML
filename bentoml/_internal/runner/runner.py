from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING
from functools import lru_cache

import attr

from ..types import ParamSpec
from ..utils import first_not_none
from .resource import Resource
from .runnable import Runnable
from .strategy import Strategy
from .strategy import DefaultStrategy
from ...exceptions import StateException
from .runner_handle import RunnerHandle
from .runner_handle import DummyRunnerHandle

if TYPE_CHECKING:
    from ..models import Model
    from .runnable import RunnableMethodConfig

T = t.TypeVar("T", bound=Runnable)
P = ParamSpec("P")
R = t.TypeVar("R")

logger = logging.getLogger(__name__)


@attr.frozen(slots=False)
class RunnerMethod(t.Generic[T, P, R]):
    runner: Runner
    name: str
    config: RunnableMethodConfig
    max_batch_size: int
    max_latency_ms: int

    def run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return self.runner._runner_handle.run_method(  # type: ignore
            self,
            *args,
            **kwargs,
        )

    async def async_run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return await self.runner._runner_handle.async_run_method(  # type: ignore
            self,
            *args,
            **kwargs,
        )


# TODO: Move these to the default configuration file and allow user override
GLOBAL_DEFAULT_MAX_BATCH_SIZE = 100
GLOBAL_DEFAULT_MAX_LATENCY_MS = 10000


@attr.define(slots=False, frozen=True, eq=False)
class Runner:
    runnable_class: t.Type[Runnable]
    runnable_init_params: t.Dict[str, t.Any]
    name: str
    models: t.List[Model]
    resource_config: Resource
    runner_methods: list[RunnerMethod[t.Any, t.Any, t.Any]]
    scheduling_strategy: t.Type[Strategy]

    _runner_handle: RunnerHandle = attr.field(init=False, factory=DummyRunnerHandle)

    def __init__(
        self,
        runnable_class: t.Type[Runnable],
        *,
        init_params: t.Dict[str, t.Any] | None = None,
        name: str | None = None,
        scheduling_strategy: t.Type[Strategy] = DefaultStrategy,
        models: t.List[Model] | None = None,
        cpu: int
        | None = None,  # TODO: support str and float type here? e.g "500m" or "0.5"
        nvidia_gpu: int | None = None,
        custom_resources: t.Dict[str, float] | None = None,
        max_batch_size: int | None = None,
        max_latency_ms: int | None = None,
        method_configs: t.Dict[str, t.Dict[str, int]] | None = None,
    ) -> None:
        """
        Args:
            runnable_class: runnable class
            init_params: runnable init params
            name: runner name
            scheduling_strategy: scheduling strategy
            models: list of required bento models
            cpu: cpu resource
            nvidia_gpu: nvidia gpu resource
            custom_resources: custom resources
            max_batch_size: max batch size config for micro batching
            max_latency_ms: max latency config for micro batching
            method_configs: per method configs
        """

        name = runnable_class.__name__ if name is None else name
        models = [] if models is None else models
        runner_method_map: dict[str, RunnerMethod[t.Any, t.Any, t.Any]] = {}
        runner_init_params = {} if init_params is None else init_params
        method_configs = {} if method_configs is None else {}
        custom_resources = {} if custom_resources is None else {}
        resource_config = Resource(
            cpu=cpu,
            nvidia_gpu=nvidia_gpu,
            custom_resources=custom_resources or {},
        )

        for method_name, method in runnable_class.methods.items():
            method_max_batch_size = method_configs.get(method_name, {}).get(
                "max_batch_size"
            )
            method_max_latency_ms = method_configs.get(method_name, {}).get(
                "max_latency_ms"
            )

            runner_method_map[method_name] = RunnerMethod(
                runner=self,
                name=method_name,
                config=method.config,
                max_batch_size=first_not_none(
                    method_max_batch_size,
                    max_batch_size,
                    default=GLOBAL_DEFAULT_MAX_BATCH_SIZE,
                ),
                max_latency_ms=first_not_none(
                    method_max_latency_ms,
                    max_latency_ms,
                    default=GLOBAL_DEFAULT_MAX_LATENCY_MS,
                ),
            )

        self.__attrs_init__(  # type: ignore
            runnable_class=runnable_class,
            runnable_init_params=runner_init_params,
            name=name,
            models=models,
            resource_config=resource_config,
            runner_methods=list(runner_method_map.values()),
            scheduling_strategy=scheduling_strategy,
        )

        # Choose the default method:
        #  1. if there's only one method, it will be set as default
        #  2. if there's a method named "__call__", it will be set as default
        #  3. otherwise, there's no default method
        if len(runner_method_map) == 1:
            default_method = next(iter(runner_method_map.values()))
            logger.info(
                f"Default runner method set to `{default_method.name}`, it can be accessed both via `runner.run` and `runner.{default_method.name}.async_run`"
            )
        elif "__call__" in runner_method_map:
            default_method = runner_method_map["__call__"]
            logger.info(
                "Default runner method set to `__call__`, it can be accessed via `runner.run` or `runner.async_run`"
            )
        else:
            default_method = None
            logger.info(
                f'No default method found for Runner "{name}", all method access needs to be in the form of `runner.{{method}}.run`'
            )

        # set default run method entrypoint
        if default_method is not None:
            object.__setattr__(self, "run", default_method.run)
            object.__setattr__(self, "async_run", default_method.async_run)

        # set all run method entrypoint
        for runner_method in self.runner_methods:
            object.__setattr__(self, runner_method.name, runner_method)

    @lru_cache(maxsize=1)
    def get_effective_resource_config(self) -> Resource:
        return (
            Resource.from_config(self.name)
            | self.resource_config
            | Resource.from_system()
        )

    def _init(self, handle_class: t.Type[RunnerHandle]) -> None:
        if not isinstance(self._runner_handle, DummyRunnerHandle):
            raise StateException("Runner already initialized")

        runner_handle = handle_class(self)
        object.__setattr__(self, "_runner_handle", runner_handle)

    def _init_local(self) -> None:
        from .runner_handle.local import LocalRunnerRef

        self._init(LocalRunnerRef)

    def init_local(self, quiet: bool = False) -> None:
        """
        init local runnable instance, for testing and debugging only
        """
        if not quiet:
            logger.warning("'Runner.init_local' is for debugging and testing only")

        self._init_local()

    def init_client(self):
        """
        init client for a remote runner instance
        """
        from .runner_handle.remote import RemoteRunnerClient

        self._init(RemoteRunnerClient)

    def destroy(self):
        object.__setattr__(self, "_runner_handle", DummyRunnerHandle())
