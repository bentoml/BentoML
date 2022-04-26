import logging
import os
import re
import sys
import enum
import typing as t
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import attr
import psutil


from bentoml import Tag
from bentoml.exceptions import BentoMLException

from .utils import cpu_converter
from .utils import gpu_converter
from .utils import mem_converter
from .utils import query_cgroup_cpu_count
from ..utils import cached_property
from ..configuration.containers import DeploymentContainer

if TYPE_CHECKING:
    import platform

    if platform.system() == "Darwin":
        from psutil._psosx import svmem
    elif platform.system() == "Linux":
        from psutil._pslinux import svmem
    else:
        from psutil._pswindows import svmem

if sys.version_info >= (3, 8):
    from typing import final
else:
    final = lambda x: x


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



class RunnableMethod:
    name: str
    batchable: bool
    batch_dim: int
    #input_spec: .. # optional
    #output_spec: .. # optional


class Runnable(ABC):
    """
    Runnable base class
    """

    @classmethod
    def method(cls, runnable_method):
        # for definition runnable methods
        # @bentoml.Runnable.method
        ...

    @classmethod
    def get_runnable_methods(cls) -> t.List[RunnableMethod]:
        ...


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

class RunnerMethod:
    runner: "Runner"
    runnable_method: RunnableMethod
    max_batch_size: int
    max_latency_ms: int

    def run(self, *args, **kwargs):
        return self.runner._run(self.runnable_method.name, *args, **kwargs)

    async def async_run(self, *args, **kwargs):
        return await self.runner._async_run(self.runnable_method.name, *args, **kwargs)

class RunnerResourceConfig:
    ...

class Runner:
    """
    TODO: add docstring
    """

    def __init__(self, runnable_class, init_params, name, strategy, models, cpu, nvidia_gpu, custom_resources, max_batch_size, max_latency_ms, runnable_method_configs: t.Dict[str, t.Dict[str, str | int]]) -> None:
        self._runnable_class = runnable_class
        self._runnable_init_params = init_params
        self._name = name
        self._strategy = strategy
        self._model = models
        self._resource_config = RunnerResourceConfig(cpu, nvidia_gpu, custom_resources)

        runnable_method_configs = runnable_method_configs or {}
        for runnable_method in self.runnable_class.get_runnable_methods():
            method_config = runnable_method_configs.get(runnable_method.name, {})
            setattr(
                self,
                runnable_method.name,
                RunnerMethod(
                    self,
                    runnable_method,
                    max_batch_size=method_config.get("max_batch_size", max_batch_size),
                    max_latency_ms=method_config.get("max_batch_size", max_latency_ms),
                )
            )

        self._runner_app_client: "RunnerClient" = None
        self._runnable: Runnable = None

    @property
    def name(self):
        return self._name

    @property
    def strategy(self):
        return self._strategy

    @property
    def models(self):
        return self._models

    @property
    def resource_config(self):
        # look up configuration
        # else return self._resource_config
        ...

    def _run(self, runner_method_name, *args, **kwargs):
        if self._runner_app_client:
            return self._runner_app_client.run(runner_method_name, *args, **kwargs)
        if self._runnable:
            return self._runnable[runner_method_name](*args, **kwargs)

        raise BentoMLException("runner not initialized")  # TODO: make this UninitializedRunnerException

    async def _async_run(self, runner_method_name, *args, **kwargs):
        if self._runner_app_client:
            return await self._runner_app_client.async_run(runner_method_name, *args, **kwargs)
        if self._runnable:
            import anyio

            # TODO(jiang): to_thread
            #return await self._runnable[runner_method_name](*args, **kwargs)

        raise BentoMLException("runner not initialized")  # TODO: make this UninitializedRunnerException

    def init_local(self):
        """
        init local runnable container, for testing and debugging only
        """
        logger.warning("for debugging and testing only")  # if not called from RunnerApp
        if self._runner_app_client:
            raise BentoMLException("TODO: ..")
        if self._runnable:
            logger.warning("re creating runnable")

        self._runnable = self._runnable_class(**self._runnable_init_params)

    def destroy_local(self):
        if not self._runnable:
            logger.warning("local runnable not found")
        else:
            del self._runnable
            # self._runnable = None # do we need this?

    def _init_remote_handle(self, host):
        """
        init runner from BentoMLContainer or environment variables
        """
        ...

    # @property
    # def default_name(self) -> str:
    #     """
    #     Return the default name of the runner. Will be used if no name is provided.
    #     """
    #     return type(self).__name__
    #
    # @abstractmethod
    # def _setup(self) -> None:
    #     ...
    #
    # def _shutdown(self) -> None:
    #     # still a hidden SDK API
    #     pass
    #
    # @property
    # def num_replica(self) -> int:
    #     return 1
    #
    # @property
    # def required_models(self) -> t.List[Tag]:
    #     return []
    #
    # @cached_property
    # @final
    # def name(self) -> str:
    #     if self._name is None:
    #         name = self.default_name
    #     else:
    #         name = self._name
    #     if not name.isidentifier():
    #         return VARNAME_RE.sub("_", name)
    #     return name
    #
    # @cached_property
    # @final
    # def resource_quota(self) -> ResourceQuota:
    #     return ResourceQuota()
    #
    # @cached_property
    # @final
    # def batch_options(self) -> BatchOptions:
    #     return BatchOptions()
    #
    # @final
    # @cached_property
    # def _impl(self) -> "RunnerImpl":
    #     return create_runner_impl(self)
    #
    # @final
    # async def async_run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
    #     return await self._impl.async_run(*args, **kwargs)
    #
    # @final
    # async def async_run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
    #     return await self._impl.async_run_batch(*args, **kwargs)
    #
    # @final
    # def run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
    #     return self._impl.run(*args, **kwargs)
    #
    # @final
    # def run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
    #     return self._impl.run_batch(*args, **kwargs)



# class RunnableContainer:
#     def __init__(self, runnable_class, runnable_init_params):
#         self._runnable_class = runnable_class
#         self._runnable_init_params = runnable_init_params
#         self.
#
#     def setup(self) -> None:
#         pass
#
#     def shutdown(self) -> None:
#         pass
#
#     @abstractmethod
#     async def async_run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
#         ...
#
#     @abstractmethod
#     async def async_run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
#         ...
#
#     @abstractmethod
#     def run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
#         ...
#
#     @abstractmethod
#     def run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
#         ...

#
# def create_runner_impl(runner: BaseRunner) -> RunnerImpl:
#     remote_runner_mapping = DeploymentContainer.remote_runner_mapping.get()
#     if runner.name in remote_runner_mapping:
#         from .remote import RemoteRunnerClient
#
#         impl = RemoteRunnerClient(runner)
#     else:
#         from .local import LocalRunner
#
#         impl = LocalRunner(runner)
#
#     return impl
