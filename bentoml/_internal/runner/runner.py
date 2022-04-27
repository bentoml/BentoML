import os
import abc
import math
import typing as t
import logging
import collections
from typing import TYPE_CHECKING

import attr

from bentoml.exceptions import BentoMLException

from .runnable import Runnable

if TYPE_CHECKING:

    class Model:
        pass

    class MethodConfig(t.TypedDict):
        max_batch_size: t.Optional[int]
        max_latency_ms: t.Optional[int]


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


class RunnerResourceConfig:
    ...


@attr.define(frozen=True)
class Resource:
    cpu: int = attr.field()
    nvidia_gpu: int = attr.field()
    custom_resources: t.Dict[str, t.Union[float, int]] = attr.field(factory=dict)


class Strategy(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def get_worker_count(
        cls,
        runnable_class: t.Type[Runnable],
        resource_request: Resource,
    ) -> int:
        ...

    @classmethod
    @abc.abstractmethod
    def setup_worker(
        cls,
        runnable_class: t.Type[Runnable],
        resource_request: Resource,
        worker_index: int,
    ) -> None:
        ...


class DefaultStrategy(Strategy):
    @classmethod
    @abc.abstractmethod
    def get_worker_count(
        cls,
        runnable_class: t.Type[Runnable],
        resource_request: Resource,
    ) -> int:
        if resource_request.nvidia_gpu > 0 and runnable_class.SUPPORT_NVIDIA_GPU:
            return math.ceil(resource_request.nvidia_gpu)

        if runnable_class.SUPPORT_MULTIPLE_CPU_THREADS:
            return 1

        return math.ceil(resource_request.cpu)

    @classmethod
    @abc.abstractmethod
    def setup_worker(
        cls,
        runnable_class: t.Type[Runnable],
        resource_request: Resource,
        worker_index: int,
    ) -> None:
        # use nvidia gpu
        if resource_request.nvidia_gpu > 0 and runnable_class.SUPPORT_NVIDIA_GPU:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_index)
            return

        # use CPU
        if runnable_class.SUPPORT_MULTIPLE_CPU_THREADS:
            thread_count = math.ceil(resource_request.cpu)
            os.environ["OMP_THREADS"] = str(thread_count)
            return

        os.environ["OMP_THREADS"] = "1"
        return


@attr.define(frozen=True)
class Runner:
    runnable_class: t.Type[Runnable] = attr.field()
    init_params: t.Dict[str, t.Any] = attr.field()
    name: str = attr.field()
    strategy: t.Type[Strategy] = attr.field(default=DefaultStrategy)
    models: t.List[Model] = attr.field(factory=list)
    resource_request: Resource = attr.field()
    method_configs = attr.field()

    def __init__(
        self,
        runnable_class: t.Type[Runnable],
        init_params,
        name,
        strategy,
        models,
        cpu,
        nvidia_gpu,
        custom_resources,
        max_batch_size: int,
        max_latency_ms: int,
        runnable_method_configs=None,
    ) -> None:
        self._runnable_class = runnable_class
        self._runnable_init_params = init_params
        self._name = name
        self._strategy = strategy
        self._model = models
        self._resource_config = Resource(cpu, nvidia_gpu, custom_resources)

        self._method_configs = collections.defaultdict(
            lambda: {"max_batch_size": max_batch_size, "max_latency_ms": max_latency_ms}
        )
        if runnable_method_configs is not None:
            self._method_configs.update(runnable_method_configs)

        for (
            name,
            runnable_method,
        ) in runnable_class.get_method_configs().items():
            setattr(self, name, RunnerMethod(self, name))

        self._runner_app_client: "RunnerClient" = None
        self._runnable: Runnable = None

    @property
    def resource_config(self):
        # look up configuration
        # else return self._resource_config
        ...

    def run_method(self, method_name: str, *args, **kwargs) -> t.Any:
        if self._runner_app_client:
            return self._runner_app_client.run(method_name, *args, **kwargs)
        if self._runnable:
            return self._runnable[method_name](*args, **kwargs)

        raise BentoMLException(
            "runner not initialized"
        )  # TODO: make this UninitializedRunnerException

    async def async_run_method(self, method_name: str, *args, **kwargs) -> t.Any:
        if self._runner_app_client:
            return await self._runner_app_client.async_run(method_name, *args, **kwargs)
        if self._runnable:
            import anyio

            # TODO(jiang): to_thread
            # return await self._runnable[runner_method_name](*args, **kwargs)

        raise BentoMLException(
            "runner not initialized"
        )  # TODO: make this UninitializedRunnerException

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

    def _init_remote_client(self, host):
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
