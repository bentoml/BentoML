import enum
import os
import typing as t
from abc import ABC, abstractmethod

import attr
import psutil
from simple_di import Provide, inject

from ..configuration.containers import BentoServerContainer
from ..runner.container import AutoContainer
from ..runner.utils import Params
from .utils import (
    _cpu_converter,
    _gpu_converter,
    _mem_converter,
    _query_cgroup_cpu_count,
)


@attr.s
class ResourceQuota:
    cpu = attr.ib(converter=_cpu_converter, type=float)
    mem = attr.ib(converter=_mem_converter, type=int)

    # Example gpus value: "all", 2, "device=1,2"
    # Default to "None", returns all available GPU devices in current environment
    gpus = attr.ib(
        converter=_gpu_converter,
        type=t.List[str],
        default=None,
    )

    @cpu.default
    def _get_default_cpu(self) -> float:
        # Default to the total CPU count available in current node or cgroup
        if psutil.POSIX:
            return _query_cgroup_cpu_count()
        else:
            return float(os.cpu_count())

    @mem.default
    def _get_default_mem(self) -> int:
        # Default to the total memory available
        from psutil import virtual_memory

        mem = virtual_memory()
        return mem.total

    @property
    def on_gpu(self) -> bool:
        if self.gpus is not None:
            return len(self.gpus) > 0
        return False


@attr.s
class BatchOptions:
    enabled = attr.ib(
        type=bool,
        default=attr.Factory(BentoServerContainer.config.batch_options.enabled.get),
    )
    max_batch_size = attr.ib(
        type=int,
        default=attr.Factory(
            BentoServerContainer.config.batch_options.max_batch_size.get
        ),
    )
    max_latency_ms = attr.ib(
        type=int,
        default=attr.Factory(
            BentoServerContainer.config.batch_options.max_latency_ms.get
        ),
    )
    input_batch_axis = attr.ib(type=int, default=0)
    output_batch_axis = attr.ib(type=int, default=0)


class _BaseRunner:
    name: str
    resource_quota: "ResourceQuota"
    batch_options: "BatchOptions"

    def __init__(
        self,
        name: str,
        resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
        batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        self.name = name
        self.resource_quota = ResourceQuota(**resource_quota if resource_quota else {})
        self.batch_options = BatchOptions(**batch_options if batch_options else {})

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
    def _setup(self, **kwargs: t.Any) -> None:
        ...

    @inject
    def _impl_ref(
        self,
        remote_runner_mapping: t.Dict[str, int] = Provide[
            BentoServerContainer.remote_runner_mapping
        ],
    ) -> "RunnerImpl":
        # TODO(jiang): cache impl
        if self.name in remote_runner_mapping:
            return RemoteRunner(self)
        else:
            return LocalRunner(self)

    async def async_run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return await self._impl_ref().async_run(*args, **kwargs)

    async def async_run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return await self._impl_ref().async_run_batch(*args, **kwargs)

    def run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return self._impl_ref().run(*args, **kwargs)

    def run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return self._impl_ref().run_batch(*args, **kwargs)


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

    def _run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
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


class RunnerState(enum.IntEnum):
    INIT = 0
    SETTING = 1
    SET = 2


class RunnerImpl:
    def __init__(self, runner: _BaseRunner):
        self._runner = runner
        self._state: RunnerState = RunnerState.INIT

    @abstractmethod
    async def async_run(self, *args, **kwargs):
        ...

    @abstractmethod
    async def async_run_batch(self, *args, **kwargs):
        ...

    @abstractmethod
    def run(self, *args, **kwargs):
        ...

    @abstractmethod
    def run_batch(self, *args, **kwargs):
        ...


class RemoteRunner(RunnerImpl):
    @property
    @inject
    def _client(
        self, remote_runner_mapping=Provide[BentoServerContainer.remote_runner_mapping]
    ):
        from .client import RunnerClient

        uds = remote_runner_mapping.get(self._runner.name)
        return RunnerClient(uds)  # TODO(jiang): timeout

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
        self._state = RunnerState.SETTING
        self._runner._setup()  # noqa
        self._state = RunnerState.SET

    async def async_run(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    async def async_run_batch(self, *args, **kwargs):
        if self._state is RunnerState.INIT:
            self._setup()
        if isinstance(self._runner, Runner):
            return self._runner._run_batch(*args, **kwargs)
        if isinstance(self._runner, SimpleRunner):
            raise RuntimeError("shall not call async_run_batch on a simple runner")

    def run(self, *args, **kwargs):
        if self._state is RunnerState.INIT:
            self._setup()
        if isinstance(self._runner, Runner):
            params = Params(*args, **kwargs)
            params = params.map(
                lambda i: AutoContainer.singles_to_batch(
                    [i], batch_axis=self._runner.batch_options.input_batch_axis
                )
            )
            batch_result = self._runner._run_batch(*params.args, **params.kwargs)
            return AutoContainer.batch_to_singles(
                batch_result,
                batch_axis=self._runner.batch_options.output_batch_axis,
            )[0]

        if isinstance(self._runner, SimpleRunner):
            return self._runner._run(*args, **kwargs)

    def run_batch(self, *args, **kwargs):
        if self._state is RunnerState.INIT:
            self._setup()
        if isinstance(self._runner, Runner):
            print(len(args), len(kwargs))
            return self._runner._run_batch(*args, **kwargs)
        if isinstance(self._runner, SimpleRunner):
            raise RuntimeError("shall not call run_batch on a simple runner")
