import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import round
from typing import Optional, Union

import attr
import psutil

from .utils import _cpu_converter, _mem_converter, _query_cpu_count


@attr.s
class RunnerResourceLimits:
    cpu = attr.ib(convert=_cpu_converter, type=float)
    mem = attr.ib(convert=_mem_converter, type=int)
    gpu = attr.ib(type=float, default=0.0)

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

    @property
    def on_gpu(self) -> bool:
        return self.gpu > 0.0


@attr.s
class RunnerBatchOptions:
    enabled = attr.ib(type=bool, default=True)
    max_batch_size = attr.ib(type=int, default=10000)
    max_latency_ms = attr.ib(type=int, default=10000)


class Runner(ABC):
    """
    Usage:
    r = bentoml.xgboost.load_runner()
    r.resource_limits.cpu = 2
    r.resource_limits.mem = "2Gi"

    class XgboostRunner(Runner):

        def __init__(self, name, model_path):
            super().__init__(name)
            self.model_path = model_path

        def _setup(self):
            self.model = load(model_path)
            ...
    """

    def __init__(self, name):
        self.name = name
        self.resource_limits = RunnerResourceLimits()
        self.batch_options = RunnerBatchOptions()

    @property
    def num_concurrency(self):
        return 1

    @property
    def num_replica(self):
        return 1

    @abstractmethod
    def _setup(self):
        ...

    @abstractmethod
    def _run_batch(self):
        ...
