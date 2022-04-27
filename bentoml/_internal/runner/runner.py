from __future__ import annotations

import typing as t
import logging
import collections
from typing import TYPE_CHECKING

import attr

from bentoml.exceptions import BentoMLException

from .remote import RemoteRunnerClient
from .runnable import Runnable
from .strategy import Strategy
from .strategy import DefaultStrategy

if TYPE_CHECKING:
    from ..models import Model


logger = logging.getLogger(__name__)


# def _get_default_cpu() -> float:
#     # Default to the total CPU count available in current node or cgroup
#     if psutil.POSIX:
#         return query_cgroup_cpu_count()
#     else:
#         cpu_count = os.cpu_count()
#         if cpu_count is not None:
#             return float(cpu_count)
#         raise ValueError("CPU count is NoneType")
#
#
# def _get_default_mem() -> int:
#     # Default to the total memory available
#     from psutil import virtual_memory
#
#     mem: "svmem" = virtual_memory()
#     return mem.total


# @attr.define
# class ResourceQuota:
#     cpu: float = attr.field(converter=cpu_converter, factory=_get_default_cpu)
#     mem: int = attr.field(converter=mem_converter, factory=_get_default_mem)
#     # Example gpus value: "all", 2, "device=1,2"
#     # Default to "None", returns all available GPU devices in current environment
#     gpus: t.List[str] = attr.field(converter=gpu_converter, default=None)
#
#     @property
#     def on_gpu(self) -> bool:
#         if self.gpus is not None:
#             return len(self.gpus) > 0
#         return False
#
#
# @attr.s
# class BatchOptions:
#     enabled = attr.ib(
#         type=bool,
#         default=attr.Factory(
#             DeploymentContainer.api_server_config.batch_options.enabled.get
#         ),
#     )
#     max_batch_size = attr.ib(
#         type=int,
#         default=attr.Factory(
#             DeploymentContainer.api_server_config.batch_options.max_batch_size.get
#         ),
#     )
#     max_latency_ms = attr.ib(
#         type=int,
#         default=attr.Factory(
#             DeploymentContainer.api_server_config.batch_options.max_latency_ms.get
#         ),
#     )
#     input_batch_axis = attr.ib(type=int, default=0)
#     output_batch_axis = attr.ib(type=int, default=0)


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


@attr.define()
class RunnerMethod:
    runner: "Runner" = attr.field()
    method_name: str = attr.field()

    def run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return self.runner.run_method(self.method_name, *args, **kwargs)

    async def async_run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return await self.runner.async_run_method(self.method_name, *args, **kwargs)


@attr.define(frozen=True)
class Resource:
    cpu: int = attr.field()
    nvidia_gpu: int = attr.field()
    custom_resources: t.Dict[str, t.Union[float, int]] = attr.field(factory=dict)


@attr.define(frozen=True)
class RunnerMethodConfig:
    max_batch_size: int = attr.field(default=1000)
    max_latency_ms: int = attr.field(default=10000)


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


@attr.define(frozen=True)
class Runner:
    runnable_class: t.Type[Runnable] = attr.field()
    init_params: t.Dict[str, t.Any] = attr.field()
    name: str = attr.field()
    strategy: t.Type[Strategy] = attr.field(default=DefaultStrategy)
    models: t.List[Model] = attr.field(factory=list)
    resource_config: Resource = attr.field()
    method_configs = attr.field()
    runner_handle = attr.field(init=False)

    def __init__(
        self,
        runnable_class: t.Type[Runnable],
        init_params: t.Dict[str, t.Any],
        name: str,
        strategy: t.Type[Strategy],
        models: t.List[Model],
        cpu: int,
        nvidia_gpu: int,
        custom_resources: t.Dict[str, int | float],
        max_batch_size: int,
        max_latency_ms: int,
        method_configs: t.Dict[str, RunnerMethodConfig] | None,
    ) -> None:
        self.__attrs_init__(  # type: ignore
            runnable_class,
            init_params,
            name,
            strategy,
            models,
            resource_config=Resource(
                cpu=cpu,
                nvidia_gpu=nvidia_gpu,
                custom_resources=custom_resources,
            ),
            method_configs=collections.defaultdict(
                lambda: RunnerMethodConfig(
                    max_batch_size=max_batch_size,
                    max_latency_ms=max_latency_ms,
                ),
                method_configs or {},
            ),
        )

        for name in runnable_class.get_method_configs().keys():
            setattr(self, name, RunnerMethod(self, name))

    def run_method(
        self,
        method_name: str,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> t.Any:
        return self.runner_handle.run_method(method_name, *args, **kwargs)

    async def async_run_method(
        self,
        method_name: str,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> t.Any:
        return await self.runner_handle.async_run_method(method_name, *args, **kwargs)

    def init_local(self):
        """
        init local runnable container, for testing and debugging only
        """
        self.runner_handle.init_local(self)

    def destroy_local(self):
        self.runner_handle.destroy_local()

    def init_client(self):
        """
        init runner from BentoMLContainer or environment variables
        """
        self.runner_handle.init_client(self)

    def destroy_client(self):
        self.runner_handle.destroy_client()
