import os
import re
import enum
import typing as t
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import attr
import psutil

from .utils import cpu_converter
from .utils import gpu_converter
from .utils import mem_converter
from .utils import query_cgroup_cpu_count
from ..types import Tag
from ..configuration.containers import BentoServerContainer

if TYPE_CHECKING:
    import platform

    if platform.system() == "Darwin":
        from psutil._psosx import svmem
    elif platform.system() == "Linux":
        from psutil._pslinux import svmem
    else:
        from psutil._pswindows import svmem


@attr.define
class ResourceQuota:
    cpu: float = attr.field(converter=cpu_converter)
    mem: int = attr.field(converter=mem_converter)

    # Example gpus value: "all", 2, "device=1,2"
    # Default to "None", returns all available GPU devices in current environment
    gpus: t.List[str] = attr.field(converter=gpu_converter, default=None)

    @cpu.default  # type: ignore
    def _get_default_cpu(self) -> float:
        # Default to the total CPU count available in current node or cgroup
        if psutil.POSIX:
            return query_cgroup_cpu_count()
        else:
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                return float(cpu_count)
            raise ValueError("CPU count is NoneType")

    @mem.default  # type: ignore
    def _get_default_mem(self) -> int:
        # Default to the total memory available
        from psutil import virtual_memory

        mem: "svmem" = virtual_memory()
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


VARNAME_RE = re.compile(r"\W|^(?=\d)")


class _BaseRunner:
    EXIST_NAMES: t.Set[str] = set()

    def __init__(
        self,
        display_name: t.Union[str, Tag],
        resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
        batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        # probe an unique name
        if isinstance(display_name, Tag):
            display_name = display_name.name
        if not display_name.isidentifier():
            display_name = VARNAME_RE.sub("_", display_name)
        i = 0
        while True:
            name = display_name if i == 0 else f"{display_name}_{i}"
            if name not in self.EXIST_NAMES:
                self.EXIST_NAMES.add(name)
                break
            else:
                i += 1
        self.name = name

        self.resource_quota = ResourceQuota(
            **(resource_quota if resource_quota else {})
        )
        self.batch_options = BatchOptions(**(batch_options if batch_options else {}))

    @property
    def num_concurrency_per_replica(self) -> int:
        return 1

    @property
    def num_replica(self) -> int:
        return 1

    @property
    def required_models(self) -> t.List[Tag]:
        return []

    @abstractmethod
    def _setup(self) -> None:
        ...

    @property
    def _impl(self) -> "RunnerImpl":
        return RunnerImplPool.get_by_runner(self)

    async def async_run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return await self._impl.async_run(*args, **kwargs)

    async def async_run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return await self._impl.async_run_batch(*args, **kwargs)

    def run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return self._impl.run(*args, **kwargs)

    def run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return self._impl.run_batch(*args, **kwargs)


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
    def _run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        ...


class RunnerState(enum.IntEnum):
    INIT = 0
    SETTING = 1
    SET = 2


class RunnerImpl:
    def __init__(self, runner: _BaseRunner):
        self._runner = runner
        self._state: RunnerState = RunnerState.INIT

    def setup(self) -> None:
        pass

    @abstractmethod
    async def async_run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        ...

    @abstractmethod
    async def async_run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        ...

    @abstractmethod
    def run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        ...

    @abstractmethod
    def run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        ...


class RunnerImplPool:
    _runner_map: t.Dict[str, RunnerImpl] = {}

    @classmethod
    def get_by_runner(cls, runner: _BaseRunner) -> RunnerImpl:
        if runner.name in cls._runner_map:
            return cls._runner_map[runner.name]

        remote_runner_mapping = BentoServerContainer.remote_runner_mapping.get()
        if runner.name in remote_runner_mapping:
            from .remote import RemoteRunnerClient

            impl = RemoteRunnerClient(runner)
        else:
            from .local import LocalRunner

            impl = LocalRunner(runner)

        cls._runner_map[runner.name] = impl
        return impl
