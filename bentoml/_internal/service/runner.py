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


USAGE_DOC = """\
Usage::
    r = bentoml.xgboost.load_runner()
    r.resource_limits.cpu = 2
    r.resource_limits.mem = "2Gi"

    # Runners config override:
    runners = {
        "my_model:any": {
            "resource_limits": {"cpu": 1},
            "batch_options": {"max_batch_size": 1000},
        },
        "runner_bar": {"resource_limits": {"cpu": "200m"}},
    }

    # bentoml.xgboost.py:
    class _XgboostRunner(Runner):

        def __init__(self, runner_name, model_path):
            super().__init__(name)

        def _setup(self):
            self.model = load(model_path)
            ...

    # model_tag example:
    #  "my_nlp_model:20210810_A23CDE", "my_nlp_model:latest"
    def load_runner(model_tag: str):
        model_info = bentoml.models.get(model_tag)
        assert model_info.module == "bentoml.xgboost"
        return _XgboostRunner(model_tag, model_info.path)

    def save(name: str, model: xgboost.Model, **save_options):
        with bentoml.models.add(
            name,
            module=__module__,
            options: save_options) as ctx:

            # ctx( path, version, metadata )
            model.save(ctx.path)
            ctx.metadata.set('param_a', 'value_b')
            ctx.metadata.set('param_foo', 'value_bar')

    def load(name: str) -> xgboost.Model:
        model_info = bentoml.models.get(model_tag)
        assert model_info.module == "bentoml.xgboost"
        return xgboost.load_model(model_info.path)

    # custom runner
    class _MyRunner(Runner):

        def _setup(self):
            self.model = load("./my_model.pt")

        def _run_batch(self, ...):
            pass
"""

RUNNER_INIT_DOCS = f"""\
Runner is a tiny computation unit that runs framework inference. Runner makes it
 easy to leverage multiple threads or processes in a Python-centric model serving
 architecture, where users can also have the ability to fine tune the performance
 per thread/processes level.
 
{USAGE_DOC}
Users can also customize options for each Runner with::
    cpu (`float`, default to `1.0`): # of CPUs for a Runner
    mem (`Union[str, int]`, default to `100Mi`): Default memory allocated for a Runner
    gpu (`float`, default to `0.0`): # of GPUs for a Runner
    enable_batch (`bool`, default to `True`): enable dynamic-batching by default
    max_batch_size (`int`, default to `10000`): maximum batch size
    max_latency_ms (`int`, default to `10000`): maximum latency before raises Error

Args:
    name (`str`):
        Model name previously saved in BentoML modelstore.
    runner_name (`Optional[str]`, default to `module_name_uuid`):
        name of given runner"""

RUNNER_RETURNS_DOCS = """\
Returns:
    Runner instances for {model}. Users can then access `run_batch` to start running inference."""


class Runner(ABC):
    def __init__(self, name: str, model_path: str):
        self.name = name
        self.model_path = model_path
        self.resource_limits = RunnerResourceLimits()
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
