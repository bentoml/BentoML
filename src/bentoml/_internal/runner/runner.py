from __future__ import annotations

import typing as t
import logging
from abc import ABC
from abc import abstractmethod

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

if t.TYPE_CHECKING:
    from ...triton import Runner as TritonRunner
    from .runnable import RunnableMethodConfig

    # only use ParamSpec in type checking, as it's only in 3.10
    P = t.ParamSpec("P")
    ListModel = list[Model]
else:
    P = t.TypeVar("P")
    ListModel = list

T = t.TypeVar("T", bound=Runnable)
R = t.TypeVar("R")


logger = logging.getLogger(__name__)

object_setattr = object.__setattr__


@attr.frozen(slots=False)
class RunnerMethod(t.Generic[T, P, R]):
    runner: Runner | TritonRunner
    name: str
    config: RunnableMethodConfig
    max_batch_size: int
    max_latency_ms: int

    def run(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.runner._runner_handle.run_method(self, *args, **kwargs)

    async def async_run(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return await self.runner._runner_handle.async_run_method(self, *args, **kwargs)


def _to_lower_name(name: str) -> str:
    lname = name.lower()
    if name != lname:
        logger.warning("Converting runner name '%s' to lowercase: '%s'", name, lname)

    return lname


def _validate_name(_: t.Any, attr: attr.Attribute[str], value: str):
    try:
        validate_tag_str(value)
    except ValueError as e:
        # TODO: link to tag validation documentation
        raise ValueError(
            f"Runner name '{value}' is not valid; it must be a valid BentoML Tag name."
        ) from e


@attr.define(slots=False, frozen=True)
class AbstractRunner(ABC):
    name: str = attr.field(converter=_to_lower_name, validator=_validate_name)
    models: list[Model] = attr.field(
        converter=attr.converters.default_if_none(factory=list),
        validator=attr.validators.deep_iterable(
            attr.validators.instance_of(Model),
            iterable_validator=attr.validators.instance_of(ListModel),
        ),
    )
    resource_config: dict[str, t.Any]
    runnable_class: type[Runnable]

    @abstractmethod
    def init_local(self, quiet: bool = False) -> None:
        """
        Initialize local runnable instance, for testing and debugging only.

        Args:
            quiet: if True, no logs will be printed
        """

    @abstractmethod
    def init_client(self):
        """
        Initialize client for a remote runner instance. To be used within API server instance.
        """


@attr.define(slots=False, frozen=True, eq=False)
class Runner(AbstractRunner):

    if t.TYPE_CHECKING:
        # This will be set by __init__. This is for type checking only.
        run: t.Callable[..., t.Any]
        async_run: t.Callable[..., t.Awaitable[t.Any]]

        # the following annotations hacks around the fact that Runner does not
        # have information about signatures at runtime.
        @t.overload
        def __getattr__(self, item: t.Literal["__attrs_init__"]) -> t.Callable[..., None]:  # type: ignore
            ...

        @t.overload
        def __getattr__(self, item: t.LiteralString) -> RunnerMethod[t.Any, P, t.Any]:
            ...

        def __getattr__(self, item: str) -> t.Any:
            ...

    runner_methods: list[RunnerMethod[t.Any, t.Any, t.Any]]
    scheduling_strategy: type[Strategy]
    runnable_init_params: dict[str, t.Any] = attr.field(
        default=None, converter=attr.converters.default_if_none(factory=dict)
    )
    _runner_handle: RunnerHandle = attr.field(init=False, factory=DummyRunnerHandle)

    def _set_handle(
        self, handle_class: type[RunnerHandle], *args: P.args, **kwargs: P.kwargs
    ) -> None:
        if not isinstance(self._runner_handle, DummyRunnerHandle):
            raise StateException("Runner already initialized")

        runner_handle = handle_class(self, *args, **kwargs)
        object_setattr(self, "_runner_handle", runner_handle)

    @inject
    async def runner_handle_is_ready(
        self,
        timeout: int = Provide[BentoMLContainer.api_server_config.runner_probe.timeout],
    ) -> bool:
        """
        Check if given runner handle is ready. This will be used as readiness probe in Kubernetes.
        """
        return await self._runner_handle.is_ready(timeout)

    def __init__(
        self,
        runnable_class: type[Runnable],
        *,
        runnable_init_params: dict[str, t.Any] | None = None,
        name: str | None = None,
        scheduling_strategy: type[Strategy] = DefaultStrategy,
        models: list[Model] | None = None,
        max_batch_size: int | None = None,
        max_latency_ms: int | None = None,
        method_configs: dict[str, dict[str, int]] | None = None,
    ) -> None:
        """

        Runner represents a unit of computation that can be executed on a remote Python worker and scales independently
        See https://docs.bentoml.org/en/latest/concepts/runner.html for more details.

        Args:
            runnable_class: Runnable class that can be executed on a remote Python worker.
            runnable_init_params: Parameters to be passed to the runnable class constructor ``__init__``.
            name: Given a name for this runner. If not provided, name will be generated from the runnable class name.
                  Note that all name will be converted to lowercase and validate to be a valid BentoML Tag name.
            scheduling_strategy: A strategy class that implements the scheduling logic for this runner. If not provided,
                                 use the default strategy. Strategy will respect ``Runnable.SUPPORTED_RESOURCES`` as well as
                                 ``Runnable.SUPPORTS_CPU_MULTI_THREADING``.
            models: An optional list composed of ``bentoml.Model`` instances.
            max_batch_size: Max batch size config for dynamic batching. If not provided, use the default value from
                            configuration.
            max_latency_ms: Max latency config for dynamic batching. If not provided, use the default value from
                            configuration.
            method_configs: A dictionary per method config for this given Runner signatures.

        Returns:
            :obj:`bentoml.Runner`: A Runner instance.
        """

        if name is None:
            name = runnable_class.__name__.lower()
            logger.warning(
                "Using lowercased runnable class name '%s' for runner.", name
            )

        runners_config = BentoMLContainer.config.runners.get()
        # If given runner is configured, then use it. Otherwise use the default configuration.
        if name in runners_config:
            config = runners_config[name]
        else:
            config = runners_config

        if models is None:
            models = []

        runner_method_map: dict[str, RunnerMethod[t.Any, t.Any, t.Any]] = {}
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

        self.__attrs_init__(
            name=name,
            models=models,
            runnable_class=runnable_class,
            runnable_init_params=runnable_init_params,
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

    def init_local(self, quiet: bool = False) -> None:
        if not quiet:
            logger.warning(
                "'Runner.init_local' is for debugging and testing only. Make sure to remove it before deploying to production."
            )

        from .runner_handle.local import LocalRunnerRef

        try:
            self._set_handle(LocalRunnerRef)
        except Exception as e:
            import traceback

            logger.error(
                "An exception occurred while instantiating runner '%s', see details below:",
                self.name,
            )
            logger.error(traceback.format_exc())

            raise e

    def init_client(
        self,
        handle_class: type[RunnerHandle] | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        if handle_class is None:
            from .runner_handle.remote import RemoteRunnerClient

            self._set_handle(RemoteRunnerClient)
        else:
            self._set_handle(handle_class, *args, **kwargs)

    def destroy(self):
        """
        Destroy the runner. This is called when the runner is no longer needed.
        Currently used under ``on_shutdown`` event of the BentoML server.
        """
        object_setattr(self, "_runner_handle", DummyRunnerHandle())

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
