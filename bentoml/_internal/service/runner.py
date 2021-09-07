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
    def on_gpu(self):
        return self.gpu > 0.0


@attr.s
class RunnerBatchOptions:
    enabled = attr.ib(type=bool, default=True)
    max_batch_size = attr.ib(type=int, default=10000)
    max_latency_ms = attr.ib(type=int, default=10000)


@attr.s
class Runner:
    name = attr.ib(type=str)

    resource_limits = attr.ib(factory=RunnerResourceLimits, init=False)
    batch_options = attr.ib(factory=RunnerBatchOptions, init=False)

    run_batch = attr.ib(type=callable)

    @run_batch.validate
    def _validate_run_batch(self):
        pass


XGBoostRunner = Runner("my_runner", run_batch=xgboost_run_batch)
