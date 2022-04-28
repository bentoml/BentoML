from __future__ import annotations

import typing as t
import logging
import collections
from typing import TYPE_CHECKING

import attr

from .remote import RemoteRunnerClient
from .resource import Resource
from .runnable import Runnable
from .runnable import RunnableMethodConfig
from .strategy import Strategy
from .strategy import DefaultStrategy
from ...exceptions import BentoMLException

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
"""

"""
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
"""

"""
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
        return self.runner.run_method(self.method_name, *args, **kwargs)

    async def async_run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return await self.runner.async_run_method(self.method_name, *args, **kwargs)


@attr.define
class RunnerHandle:
    _runnable: Runnable | None = attr.field(init=False, default=None)
    _runner_client: RemoteRunnerClient | None = attr.field(init=False, default=None)

    def init_local(self, runner: Runner):
        logger.warning("for debugging and testing only")  # if not called from RunnerApp
        if self._runner_client:
            raise BentoMLException("TODO: ..")
        if self._runnable:
            logger.warning("re creating runnable")

        self._runnable = runner.runnable_class()

    def destroy_local(self):
        if not self._runnable:
            logger.warning("local runnable not found")
        else:
            del self._runnable

    def init_client(self, runner: Runner):
        if self._runner_client:
            logger.warning("re creating remote runner client")
        if self._runnable:
            raise BentoMLException("TODO: ..")

        self._runner_client = RemoteRunnerClient(runner)

    def destroy_client(self):
        if not self._runner_client:
            logger.warning("remote runner client not found")
        else:
            del self._runner_client

    def run_method(
        self,
        method_name: str,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> t.Any:
        if self._runnable is not None:
            return getattr(self._runnable, method_name)(*args, **kwargs)
        if self._runner_client is not None:
            return self._runner_client.run_method(method_name, *args, **kwargs)
        raise BentoMLException(
            "runner not initialized"
        )  # TODO: make this UninitializedRunnerException

    async def async_run_method(
        self,
        method_name: str,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> t.Any:
        if self._runnable is not None:
            import anyio

            method = getattr(self._runnable, method_name)
            return anyio.to_thread.run_sync(method, *args, **kwargs)
        if self._runner_client is not None:
            return await self._runner_client.async_run_method(
                method_name,
                *args,
                **kwargs,
            )
        raise BentoMLException(
            "runner not initialized"
        )  # TODO: make this UninitializedRunnerException


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
    _runner_handle = attr.field(init=False)

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
        custom_resources: t.Dict[str, int | float] | None = None,
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
        runner_methods = []
        method_configs = method_configs or {}
        runnable_method_config_map = runnable_class.get_method_configs()
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

        runner_name = name or runnable_class.__name__
        self.__attrs_init__(  # type: ignore
            runnable_class=runnable_class,
            runnable_init_params=init_params or {},
            name=runner_name,
            models=models or [],
            resource_config=Resource(
                cpu=cpu,
                nvidia_gpu=nvidia_gpu,
                custom_resources=custom_resources,
            ).with_config_overrides(runner_name),
            runner_methods=runner_methods,
            scheduling_strategy=scheduling_strategy,
        )

        for runner_method in self.runner_methods:
            if runner_method.method_name == "__call__":
                setattr(self, "run", runner_method.run)
                setattr(self, "async_run", runner_method.async_run)
            else:
                setattr(self, runner_method.method_name, runner_method)

        self._runner_handle = RunnerHandle()

    def run_method(
        self,
        method_name: str,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> t.Any:
        return self._runner_handle.run_method(method_name, *args, **kwargs)

    async def async_run_method(
        self,
        method_name: str,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> t.Any:
        return await self._runner_handle.async_run_method(method_name, *args, **kwargs)

    def init_local(self):
        """
        init local runnable instance, for testing and debugging only
        """
        self._runner_handle.init_local(self)

    def destroy_local(self):
        self._runner_handle.destroy_local()

    def init_client(self):
        """
        init client for a remote runner instance
        """
        self._runner_handle.init_client(self)

    def destroy_client(self):
        self._runner_handle.destroy_client()
