import os
import typing as t
from abc import ABC, abstractmethod

import attr
import psutil

from .utils import _cpu_converter, _mem_converter, _query_cpu_count

if t.TYPE_CHECKING:  # pylint: disable=unused-import # pragma: no cover
    import numpy as np
    import pandas as pd

_T = t.TypeVar("_T", bound=t.Union[t.List, "np.array", "pd.DataFrame"])


@attr.s
class RunnerResourceLimits:
    cpu = attr.ib(converter=_cpu_converter, type=float)
    mem = attr.ib(converter=_mem_converter, type=int)
    gpu = attr.ib(type=float)

    @cpu.default
    def _get_default_cpu(self) -> float:
        if psutil.POSIX:
            return _query_cpu_count()
        else:
            return float(os.cpu_count())

    @mem.default
    def _get_default_mem(self) -> int:
        from psutil import virtual_memory

        mem = virtual_memory()
        return mem.total

    @gpu.default
    def _get_default_gpu(self) -> float:
        # TODO:
        return 0.0

    @property
    def on_gpu(self) -> bool:
        return self.gpu > 0.0


@attr.s
class RunnerBatchOptions:
    enabled = attr.ib(type=bool, default=True)
    max_batch_size = attr.ib(type=int, default=10000)
    max_latency_ms = attr.ib(type=int, default=10000)


class Runner(ABC):
    def __init__(self, name: str, resource_quota=None, batch_options=None):
        self.name = name
        self.resource_quota = RunnerResourceLimits()
        self.batch_options = RunnerBatchOptions()

    # fmt: off
    @property
    @abstractmethod
    def num_concurrency(self): return 1

    @property
    @abstractmethod
    def num_replica(self): return 1

    @abstractmethod
    def _setup(self, *args, **kwargs): ...

    @abstractmethod
    def _run_batch(self, input_data: "_T") -> "_T": ...
    # fmt: on
