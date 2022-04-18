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

from ..tag import Tag
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


def _get_default_cpu() -> float:
    # Default to the total CPU count available in current node or cgroup
    if psutil.POSIX:
        return query_cgroup_cpu_count()
    else:
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            return float(cpu_count)
        raise ValueError("CPU count is NoneType")


def _get_default_mem() -> int:
    # Default to the total memory available
    from psutil import virtual_memory

    mem: "svmem" = virtual_memory()
    return mem.total


@attr.define
class ResourceQuota:
    cpu: float = attr.field(converter=cpu_converter, factory=_get_default_cpu)
    mem: int = attr.field(converter=mem_converter, factory=_get_default_mem)
    # Example gpus value: "all", 2, "device=1,2"
    # Default to "None", returns all available GPU devices in current environment
    gpus: t.List[str] = attr.field(converter=gpu_converter, default=None)

    @property
    def on_gpu(self) -> bool:
        if self.gpus is not None:
            return len(self.gpus) > 0
        return False


@attr.s
class BatchOptions:
    enabled = attr.ib(
        type=bool,
        default=attr.Factory(
            DeploymentContainer.api_server_config.batch_options.enabled.get
        ),
    )
    max_batch_size = attr.ib(
        type=int,
        default=attr.Factory(
            DeploymentContainer.api_server_config.batch_options.max_batch_size.get
        ),
    )
    max_latency_ms = attr.ib(
        type=int,
        default=attr.Factory(
            DeploymentContainer.api_server_config.batch_options.max_latency_ms.get
        ),
    )
    input_batch_axis = attr.ib(type=int, default=0)
    output_batch_axis = attr.ib(type=int, default=0)


VARNAME_RE = re.compile(r"\W|^(?=\d)")


class BaseRunner:
    """
    This class should not be implemented directly. Instead, implement the SimpleRunner or Runner.
    """

    def __init__(self, name: t.Optional[str]) -> None:
        self._name = name

    @property
    def default_name(self) -> str:
        """
        Return the default name of the runner. Will be used if no name is provided.
        """
        return type(self).__name__

    @abstractmethod
    def _setup(self) -> None:
        ...

    def _shutdown(self) -> None:
        # still a hidden SDK API
        pass

    @property
    def num_replica(self) -> int:
        return 1

    @property
    def required_models(self) -> t.List[Tag]:
        return []

    @cached_property
    @final
    def name(self) -> str:
        if self._name is None:
            name = self.default_name
        else:
            name = self._name
        if not name.isidentifier():
            return VARNAME_RE.sub("_", name)
        return name

    @cached_property
    @final
    def resource_quota(self) -> ResourceQuota:
        return ResourceQuota()

    @cached_property
    @final
    def batch_options(self) -> BatchOptions:
        return BatchOptions()

    @final
    @cached_property
    def _impl(self) -> "RunnerImpl":
        return create_runner_impl(self)

    @final
    async def async_run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return await self._impl.async_run(*args, **kwargs)

    @final
    async def async_run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return await self._impl.async_run_batch(*args, **kwargs)

    @final
    def run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return self._impl.run(*args, **kwargs)

    @final
    def run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return self._impl.run_batch(*args, **kwargs)


class Runner(BaseRunner, ABC):
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

    - numpy.ndarray, pandas.DataFrame, pandas.Series, List[PickleSerializable]

    - Tuple of the types above, indicating multiple return values

    Runner `run` accepts argument value of the following types:

    - numpy.ndarray => numpy.ndarray

    - pandas.DataFrame, pandas.Series => pandas.DataFrame

    - any => List[PickleSerializable]

    Note: for pandas.DataFrame and List, the batch_axis must be 0
    """

    @abstractmethod
    def _run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        ...


class SimpleRunner(BaseRunner, ABC):
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
    SETTING_UP = 1
    READY = 2
    SHUTIING_DOWN = 3
    SHUTDOWN = 4


class RunnerImpl:
    def __init__(self, runner: BaseRunner):
        self._runner = runner
        self._state: RunnerState = RunnerState.INIT

    def setup(self) -> None:
        pass

    def shutdown(self) -> None:
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


def create_runner_impl(runner: BaseRunner) -> RunnerImpl:
    remote_runner_mapping = DeploymentContainer.remote_runner_mapping.get()
    if runner.name in remote_runner_mapping:
        from .remote import RemoteRunnerClient

        impl = RemoteRunnerClient(runner)
    else:
        from .local import LocalRunner

        impl = LocalRunner(runner)

    return impl
