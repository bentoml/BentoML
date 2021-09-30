import os
import typing as t
from abc import ABC, abstractmethod

import attr
import psutil
from simple_di import Provide, inject

from bentoml._internal.configuration.containers import BentoMLContainer

from .utils import (
    _cpu_converter,
    _gpu_converter,
    _mem_converter,
    _query_cgroup_cpu_count,
)

# bentoml serve bento.py:svc --gpus 1
# BENTOML_GPUS=1 bentoml serve bento.py:svc


@attr.s
class ResourceQuota:
    cpu = attr.ib(converter=_cpu_converter, type=float)
    mem = attr.ib(converter=_mem_converter, type=int)

    # The gpu device used will be set by BentoServer via an environment variable
    # gpus:
    #   "all", 2, "device=1,2"
    gpus = attr.ib(
        converter=_gpu_converter,
        type=t.List[str],
        default=attr.Factory(list),
    )

    @cpu.default
    def _get_default_cpu(self) -> float:
        if psutil.POSIX:
            return _query_cgroup_cpu_count()
        else:
            return float(os.cpu_count())

    @mem.default
    def _get_default_mem(self) -> int:
        from psutil import virtual_memory

        mem = virtual_memory()
        return mem.total

    @property
    def on_gpu(self) -> bool:
        return len(self.gpus) > 0


@attr.s
class BatchOptions:
    enabled = attr.ib(
        type=bool,
        default=attr.Factory(
            BentoMLContainer.config.bento_server.batch_options.enabled.get
        ),
    )
    max_batch_size = attr.ib(
        type=int,
        default=attr.Factory(
            BentoMLContainer.config.bento_server.batch_options.max_batch_size.get
        ),
    )
    max_latency_ms = attr.ib(
        type=int,
        default=attr.Factory(
            BentoMLContainer.config.bento_server.batch_options.max_latency_ms.get
        ),
    )
    input_batch_axis = attr.ib(type=int, default=0)
    output_batch_axis = attr.ib(type=int, default=0)


class _RunnerImplMixin:
    @t.overload
    def _impl_ref(
        self, deployment_type: str = Provide[BentoMLContainer.deployment_type]
    ) -> "_RunnerImplMixin":
        ...

    @inject
    def _impl_ref(
        self, deployment_type: str = Provide[BentoMLContainer.deployment_type]
    ) -> "RunnerImpl":
        # TODO(jiang): cache impl
        if deployment_type == "local":
            return LocalRunner(self)
        else:
            return RemoteRunner(self)

    async def async_run(self, *args, **kwargs):
        return await self._impl_ref().async_run(*args, **kwargs)

    async def async_run_batch(self, *args, **kwargs):
        return await self._impl_ref().async_run_batch(*args, **kwargs)

    def run(self, *args, **kwargs):
        return self._impl_ref().run(*args, **kwargs)

    def run_batch(self, *args, **kwargs):
        return self._impl_ref().run_batch(*args, **kwargs)


class _BaseRunner(_RunnerImplMixin, ABC):
    name: str
    resource_quota: "ResourceQuota"
    batch_options: "BatchOptions"

    def __init__(
        self,
        name: str,
        resource_quota: t.Dict[str, t.Any] = None,
        batch_options: t.Dict[str, t.Any] = None,
    ):
        self.name = name
        self.resource_quota = (
            ResourceQuota(**resource_quota)
            if resource_quota is not None
            else ResourceQuota()
        )
        self.batch_options = (
            BatchOptions(**batch_options)
            if batch_options is not None
            else BatchOptions()
        )

    @property
    def gpu_device_ids(self) -> t.List[str]:
        return os.environ.get("_BENTOML_RUNNER_GPU_DEVICE_IDS", "").split(",")

    @property
    def num_concurrency_per_replica(self) -> int:
        return 1

    @property
    def num_replica(self) -> int:
        return 1

    @property
    def required_models(self) -> t.List[str]:
        return []

    @abstractmethod
    def _setup(self) -> None:
        ...


class Runner(_BaseRunner, ABC):
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. This Runner class is an abstract class, used for creating
    actual Runners, by implementing `__init__`, `_setup` and `_run_batch` method.

    Runner instances exposes `run` and `run_batch` method, which will eventually be
    piped to the `_run_batch` implementation of the Runner. BentoML applies dynamic
    batching optimization for all Runners by default.

    Why use Runner:
    - Runner allow BentoML to better leverage multiple threads or processes for higher
        hardware utilization (CPU, GPU, Memory)
    - Runner enables higher concurrency in serving workload, which minimizes latency:
        you may prefetch data while model is being executed, parallelize data
        extraction, data transformation or multiple runner execution.
    - Runner comes with dynamic batching optimization, which groups `run` calls into
        batch execution when serving online, the batch size and wait time is adaptive
        to the workload. This can bring massive throughput improvement to a ML service.

    All `_run_batch` argument value must be one of the three types below:
        numpy.ndarray, pandas.DataFrame, List[PickleSerializable]

    Return value of `_run_batch` acceptable types :
        numpy.ndarray, pandas.DataFrame, pandas.Series, List[PickleSerializable]
        Or Tuple of the types above, indicating multiple return values

    Runner `run` accepts argument value of the following types:
        numpy.ndarray => numpy.ndarray
        pandas.DataFrame, pandas.Series => pandas.DataFrame
        any => List[PickleSerializable]

    Note: for pandas.DataFrame and List, the batch_axis must be 0
    """

    @abstractmethod
    def _run_batch(self, *args, **kwargs):
        ...


class SimpleRunner(_BaseRunner, ABC):
    """
    SimpleRunner is a special type of Runner that does not support dynamic batching.
    Instead of `_run_batch` in Runner, a `_run` method is expected to be defined in its
    subclasses.

    A SimpleRunner only exposes `run` method to its users.
        `SimpleRunner._run` can accept arbitrary input type that are pickle-serializable
    """

    @abstractmethod
    def _run(self, *args, **kwargs):
        ...


class RunnerImpl:
    @t.overload
    def __init__(self, runner: "_RunnerImplMixin"):
        ...  # pylint:disable=redefined-builtin

    def __init__(self, runner: t.Union[Runner, SimpleRunner]):
        self._runner = runner

    @abstractmethod
    async def run(self, *args, **kwargs):
        ...

    @abstractmethod
    async def run_batch(self, *args, **kwargs):
        ...


class RemoteRunner(RunnerImpl):
    @property
    def _client(self):
        from bentoml._internal.server.runner_client import get_runner_client

        return get_runner_client(self._runner.name)

    async def async_run(self, *args, **kwargs):
        return await self._client.async_run(*args, **kwargs)

    async def async_run_batch(self, *args, **kwargs):
        return await self._client.async_run_batch(*args, **kwargs)

    def run(self, *args, **kwargs):
        return self._client.run(*args, **kwargs)

    def run_batch(self, *args, **kwargs):
        return self._client.run_batch(*args, **kwargs)


class LocalRunner(RunnerImpl):
    def _setup(self) -> None:
        self._runner._setup()  # noqa

    async def async_run(self, *args, **kwargs):
        if isinstance(self._runner, Runner):
            params = Params(args, kwargs)
            params.apply(single_to_batch)

            bresult = self._runner._run_batch(*params.args, **params.kwargs)
            return batch_to_single(bresult)

        if isinstance(self._runner, SimpleRunner):
            return self._runner._run(*args, **kwargs)

    async def async_run_batch(self, *args, **kwargs):
        if isinstance(self._runner, Runner):
            return self._runner._run_batch(*args, **kwargs)
        if isinstance(self._runner, SimpleRunner):
            results = []
            params = Params(args, kwargs)
            for iparams in params.iter(batch_to_single_list):
                results.append(self._runner._run(*iparams.args, **iparams.kwargs))
            return single_list_to_batch(results)

    def run(self, *args, **kwargs):
        ...

    def run_batch(self, *args, **kwargs):
        ...
