from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING

import attr

from .resource import Resource
from .runnable import Runnable
from .runnable import RunnableMethodConfig
from .strategy import Strategy
from .strategy import DefaultStrategy
from ...exceptions import StateException
from .runner_handle import RunnerHandle
from .runner_handle import DummyRunnerHandle

if TYPE_CHECKING:
    from ..models import Model


logger = logging.getLogger(__name__)


"""
runners config:

runners:
  - name: iris_clf
    cpu: 4
    nvidia_gpu: 0  # requesting 0 GPU
    max_batch_size: 20
  - name: my_custom_runner
	cpu: 2
	nvidia_gpu: 2  # requesting 2 GPUs
	runnable_method_configs:
      - name: "predict"
		max_batch_size: 10
		max_latency_ms: 500


Runner API usage:

my_runner = bentoml.Runner(
	MyRunnable,
	init_params={"foo": foo, "bar": bar},
	name="custom_runner_name",
	strategy=None, # default strategy will be selected depending on the SUPPORT_GPU and SUPPORT_MULTI_THREADING flag on runnable
	models=[..],

	# below are also configurable via config file:

	# default configs:
	cpu=4,
	nvidia_gpu=1
	custom_resources={..} # reserved API for supporting custom accelerators, a custom scheduling strategy will be needed to support new hardware types
	max_batch_size=..  # default max batch size will be applied to all run methods, unless override in the runnable_method_configs
	max_latency_ms=.. # default max latency will be applied to all run methods, unless override in the runnable_method_configs

	runnable_method_configs=[
		{
			method_name="predict",
			max_batch_size=..,
			max_latency_ms=..,
		}
	],
)



Testing Runner script:

my_runner = benotml.pytorch.get(tag).to_runner(..)

os.environ["CUDA_VISIBLE_DEVICES"] = None  # if needed
my_runner.init_local()
# warning user: for testing purpose only
# warning user: resource configs are not respected

my_runner.predict.run( test_input_df )

"""


@attr.define(frozen=True)
class RunnerMethod:
    runner: Runner = attr.field()
    method_name: str = attr.field()
    runnable_method_config: RunnableMethodConfig
    max_batch_size: int
    max_latency_ms: int

    def run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return self.runner._runner_handle.run_method(  # type: ignore
            self.method_name,
            *args,
            **kwargs,
        )

    async def async_run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return await self.runner._runner_handle.async_run_method(  # type: ignore
            self.method_name,
            *args,
            **kwargs,
        )


# TODO: Move these to the default configuration file and allow user override
GLOBAL_DEFAULT_MAX_BATCH_SIZE = 100
GLOBAL_DEFAULT_MAX_LATENCY_MS = 10000


@attr.define(frozen=True)
class Runner:
    runnable_class: t.Type[Runnable]
    runnable_init_params: t.Dict[str, t.Any]
    name: str
    models: t.List[Model]
    resource_config: Resource
    runner_methods: t.List[RunnerMethod]
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
        cpu: int | None,  # TODO: support str and float type here? e.g "500m" or "0.5"
        nvidia_gpu: int | None,
        custom_resources: t.Dict[str, float | None] | None = None,
        max_batch_size: int | None,
        max_latency_ms: int | None,
        method_configs: t.Dict[str, t.Dict[str, int]] | None,
    ) -> None:
        """
        TODO: add docstring
        Args:
            runnable_class:
            init_params:
            name:
            scheduling_strategy:
            models:
            cpu:
            nvidia_gpu:
            custom_resources:
            max_batch_size:
            max_latency_ms:
            method_configs:
        """
        name = runnable_class.__name__ if name is None else name
        models = [] if models is None else models
        runner_methods: list[RunnerMethod] = []
        runner_init_params = {} if init_params is None else init_params
        method_configs = {} if method_configs is None else {}
        custom_resources = {} if custom_resources is None else {}
        runnable_method_config_map = runnable_class.get_method_configs()
        resource = (
            Resource.from_config()
            | Resource(
                cpu=cpu,
                nvidia_gpu=nvidia_gpu,
                custom_resources=custom_resources or {},
            )
            | Resource.from_system()
        )

        for method_name, runnable_method_config in runnable_method_config_map.items():
            method_max_batch_size = max_batch_size or GLOBAL_DEFAULT_MAX_BATCH_SIZE
            method_max_latency_ms = max_latency_ms or GLOBAL_DEFAULT_MAX_LATENCY_MS
            if method_name in method_configs:
                if "max_batch_size" in method_configs[method_name]:
                    method_max_batch_size = method_configs[method_name][
                        "max_batch_size"
                    ]
                if "max_latency_ms" in method_configs[method_name]:
                    method_max_latency_ms = method_configs[method_name][
                        "max_latency_ms"
                    ]
                # TODO: apply user runner configs here

            runner_methods.append(
                RunnerMethod(
                    runner=self,
                    method_name=method_name,
                    runnable_method_config=runnable_method_config,
                    max_batch_size=method_max_batch_size,
                    max_latency_ms=method_max_latency_ms,
                )
            )

        self.__attrs_init__(  # type: ignore
            runnable_class=runnable_class,
            runnable_init_params=runner_init_params,
            name=name,
            models=models,
            resource_config=resource,
            runner_methods=runner_methods,
            scheduling_strategy=scheduling_strategy,
        )

        for runner_method in self.runner_methods:
            if runner_method.method_name == "__call__":
                setattr(self, "run", runner_method.run)
                setattr(self, "async_run", runner_method.async_run)
            else:
                setattr(self, runner_method.method_name, runner_method)

    def _init(self, handle_class: t.Type[RunnerHandle]) -> None:
        if not isinstance(self._runner_handle, DummyRunnerHandle):
            raise StateException("Runner already initialized")

        runner_handle = handle_class(self)
        object.__setattr__(self, "_runner_handle", runner_handle)

    def init_local(self, quiet: bool = False) -> None:
        """
        init local runnable instance, for testing and debugging only
        """
        if quiet:
            logger.warning("for debugging and testing only")

        from .runner_handle.local import LocalRunnerRef

        self._init(LocalRunnerRef)

    def init_client(self):
        """
        init client for a remote runner instance
        """
        from .runner_handle.remote import RemoteRunnerClient

        self._init(RemoteRunnerClient)

    def destroy(self):
        object.__setattr__(self, "_runner_handle", DummyRunnerHandle())
