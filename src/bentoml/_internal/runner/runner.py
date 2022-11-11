from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING

import attr
from simple_di import inject
from simple_di import Provide

from ..tag import validate_tag_str
from ..utils import first_not_none
from .runnable import Runnable
from .strategy import Strategy
from .strategy import DefaultStrategy
from ...exceptions import StateException
from ..models.model import Model
from .runner_handle import RunnerHandle
from .runner_handle import DummyRunnerHandle
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from .runnable import RunnableMethodConfig

    # only use ParamSpec in type checking, as it's only in 3.10
    P = t.ParamSpec("P")
else:
    P = t.TypeVar("P")

T = t.TypeVar("T", bound=Runnable)
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


@attr.define(slots=False, frozen=True, eq=False)
class Runner:
    runnable_class: t.Type[Runnable]
    runnable_init_params: dict[str, t.Any]
    name: str
    models: list[Model]
    resource_config: dict[str, t.Any]
    runner_methods: list[RunnerMethod[t.Any, t.Any, t.Any]]
    scheduling_strategy: t.Type[Strategy]

    _runner_handle: RunnerHandle = attr.field(init=False, factory=DummyRunnerHandle)

    if TYPE_CHECKING:
        run: t.Callable[..., t.Any]
        async_run: t.Callable[..., t.Awaitable[t.Any]]

    def __init__(
        self,
        runnable_class: t.Type[Runnable],
        *,
        runnable_init_params: t.Dict[str, t.Any] | None = None,
        name: str | None = None,
        scheduling_strategy: t.Type[Strategy] = DefaultStrategy,
        models: t.List[Model] | None = None,
        max_batch_size: int | None = None,
        max_latency_ms: int | None = None,
        method_configs: t.Dict[str, t.Dict[str, int]] | None = None,
    ) -> None:
        """
        Args:
            runnable_class: runnable class
            runnable_init_params: runnable init params
            name: runner name
            scheduling_strategy: scheduling strategy
            models: list of required bento models
            max_batch_size: max batch size config for micro batching
            max_latency_ms: max latency config for micro batching
            method_configs: per method configs
        """

        if name is None:
            lname = runnable_class.__name__.lower()
            logger.warning(
                "Using lowercased runnable class name '%s' for runner.", lname
            )
        else:
            lname = name.lower()

            if name != lname:
                logger.warning(
                    "Converting runner name '%s' to lowercase: '%s'", name, lname
                )

        try:
            validate_tag_str(lname)
        except ValueError as e:
            # TODO: link to tag validation documentation
            raise ValueError(
                f"Runner name '{name}' is not valid; it must be a valid BentoML Tag name."
            ) from e

        runners_config = BentoMLContainer.config.runners.get()
        if name in runners_config:
            config = runners_config[name]
        else:
            config = runners_config
        if models is None:
            models = []
        else:
            if not all(isinstance(model, Model) for model in models):
                raise ValueError(
                    f"models must be a list of 'bentoml.Model'. Got { {type(model) for model in models if isinstance(model, Model)} } instead."
                )
        runner_method_map: dict[str, RunnerMethod[t.Any, t.Any, t.Any]] = {}
        runnable_init_params = (
            {} if runnable_init_params is None else runnable_init_params
        )
        method_configs = {} if method_configs is None else method_configs

        if runnable_class.bentoml_runnable_methods__ is None:
            raise ValueError(
                f"Runnable class '{runnable_class.__name__}' has no methods!"
            )

        for method_name, method in runnable_class.bentoml_runnable_methods__.items():
            if not config["batching"]["enabled"]:
                method.config.batchable = False

            method_max_batch_size = None
            method_max_latency_ms = None
            if method_name in method_configs:
                method_max_batch_size = method_configs[method_name].get(
                    "max_batch_size"
                )
                method_max_latency_ms = method_configs[method_name].get(
                    "max_latency_ms"
                )

            runner_method_map[method_name] = RunnerMethod(
                runner=self,
                name=method_name,
                config=method.config,
                max_batch_size=first_not_none(
                    method_max_batch_size,
                    max_batch_size,
                    default=config["batching"]["max_batch_size"],
                ),
                max_latency_ms=first_not_none(
                    method_max_latency_ms,
                    max_latency_ms,
                    default=config["batching"]["max_latency_ms"],
                ),
            )

        self.__attrs_init__(  # type: ignore
            runnable_class=runnable_class,
            runnable_init_params=runnable_init_params,
            name=lname,
            models=models,
            resource_config=config["resources"],
            runner_methods=list(runner_method_map.values()),
            scheduling_strategy=scheduling_strategy,
        )

        # Choose the default method:
        #  1. if there's only one method, it will be set as default
        #  2. if there's a method named "__call__", it will be set as default
        #  3. otherwise, there's no default method
        if len(runner_method_map) == 1:
            default_method = next(iter(runner_method_map.values()))
            logger.debug(
                "Default runner method set to '%s', it can be accessed both via 'runner.run' and 'runner.%s.async_run'.",
                default_method.name,
                default_method.name,
            )
        elif "__call__" in runner_method_map:
            default_method = runner_method_map["__call__"]
            logger.debug(
                "Default runner method set to '__call__', it can be accessed via 'runner.run' or 'runner.async_run'."
            )
        else:
            default_method = None
            logger.debug(
                "No default method found for Runner '%s', all method access needs to be in the form of 'runner.{method}.run'.",
                name,
            )

        # set default run method entrypoint
        if default_method is not None:
            object.__setattr__(self, "run", default_method.run)
            object.__setattr__(self, "async_run", default_method.async_run)

        # set all run method entrypoint
        for runner_method in self.runner_methods:
            object.__setattr__(self, runner_method.name, runner_method)

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
            logger.warning(
                "'Runner.init_local' is for debugging and testing only. Make sure to remove it before deploying to production."
            )

        self._init_local()

    def init_client(self):
        """
        init client for a remote runner instance
        """
        from .runner_handle.remote import RemoteRunnerClient

        self._init(RemoteRunnerClient)

    def destroy(self):
        object.__setattr__(self, "_runner_handle", DummyRunnerHandle())

    @inject
    async def runner_handle_is_ready(
        self,
        timeout: int = Provide[BentoMLContainer.api_server_config.runner_probe.timeout],
    ) -> bool:
        return await self._runner_handle.is_ready(timeout)

    @property
    def scheduled_worker_count(self) -> int:
        return self.scheduling_strategy.get_worker_count(
            self.runnable_class,
            self.resource_config,
        )

    @property
    def scheduled_worker_env_map(self) -> dict[int, dict[str, t.Any]]:
        return {
            worker_id: self.scheduling_strategy.get_worker_env(
                self.runnable_class,
                self.resource_config,
                worker_id,
            )
            for worker_id in range(self.scheduled_worker_count)
        }
