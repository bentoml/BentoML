import abc
import os
from typing import Dict, Optional, Sequence, Union

import attr
import psutil
from simple_di import Provide, inject

from bentoml._internal.configuration.containers import BentoMLContainer

from .utils import _cpu_converter, _mem_converter


class Params:
    def __init__(self, *args: type, **kwargs: type):
        self.args = args
        self.kwargs = kwargs

    def apply(self, function, inplace=False):
        return type(self)(
            *tuple(function(a) for a in self.args),
            **{k: function(v) for k, v in self.kwargs.items()}
        )


class Signature:
    params: Params
    returns: type


@attr.s
class ResourceQuota:
    cpu = attr.ib(converter=_cpu_converter, type=Optional[float])
    ram = attr.ib(converter=_mem_converter, type=Optional[int])

    gpus = attr.ib(type=Union[None, int, str])
    gpu_fraction = attr.ib(type=Union[None, float, Dict[str, float]])

    @cpu.default
    def _get_default_cpu(self) -> float:
        if psutil.POSIX:
            return _query_cpu_count()
        else:
            return float(os.cpu_count())

    @ram.default
    def _get_default_ram(self) -> int:
        from psutil import virtual_memory

        mem = virtual_memory()
        return mem.total

    @gpu.default
    def _get_default_gpu(self) -> float:
        # TODO:
        return 0.0

    @property
    def gpu_device_id(self) -> int:
        return 0


@attr.s
class BatchOptions:
    enabled = attr.ib(type=bool, default=True)
    max_batch_size = attr.ib(type=int, default=10000)
    max_latency_ms = attr.ib(type=int, default=10000)
    batch_axis = 0


class _RunnerImplMixin:
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
        return await self._impl_ref().run(*args, **kwargs)

    async def rasync_run_batch(self, *args, **kwargs):
        return await self._impl_ref().run_batch(*args, **kwargs)


class _BaseRunner(_RunnerImplMixin, abc.ABC):
    name: str
    signatures: Signature
    resource_limit: ResourceQuota
    batch_options: BatchOptions

    @abc.abstractproperty
    def num_concurrency(self) -> Optional[int]:
        ...

    @abc.abstractproperty
    def num_replica(self) -> int:
        ...

    @abc.abstractproperty
    def gpu_device_mapping(self) -> Union[None, Sequence[str], Dict[str, float]]:
        """
        ["1", "2"]
        {"1": 0.1, "2": 0.1}
        """
        ...

    @abc.abstractmethod
    def _setup(self) -> None:
        ...


class Runner(_BaseRunner, abc.ABC):
    """
    Usage:
    r = bentoml.xgboost.load_runner()
    r.resource_limits.cpu = 2
    r.resource_limits.mem = "2Gi"

    Runners config override:
    "runners": {
        "my_model:any": {
            "resource_limits": {
                "cpu": 1
            },
            "batch_options": {
                "max_batch_size": 1000
            }
        }
        "runner_bar": {
            "resource_limits": {
                "cpu": 200m
            }
        }
    }

    # bentoml.xgboost.py:
    class _XgboostRunner(Runner):

        def __init__(self, runner_name, model_path):
            super().__init__(name)
            self.model_path = model_path

        def _setup(self):
            self.model = load(model_path)
            ...

    # model_tag example:
    #   "my_nlp_model:20210810_A23CDE", "my_nlp_model:latest"
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

    @abc.abstractmethod
    def _run_batch(self, *args, **kwargs):
        ...


class SimpleRunner(_BaseRunner):
    @abc.abstractmethod
    def _run(self, *args, **kwargs):
        ...


class RunnerImpl:
    def __init__(self, runner: Union[Runner, SimpleRunner]):
        self._runner = runner

    @abc.abstractmethod
    async def run(self, *args, **kwargs):
        ...

    @abc.abstractmethod
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


class LocalRunner(RunnerImpl):
    def _setup(self):
        self._runner._setup()

    async def async_run(self, **kwargs):
        if isinstance(self._runner, Runner):
            params = Params(args, kwargs)
            params = params.map(single_to_batch)

            bresult = self._runner._run_batch(**params.kwargs)
            return batch_to_single(bresult)

        if isinstance(self._runner, SimpleRunner):
            return self._runner._run(*args, **kwargs)

    async def async_run_batch(self, **kwargs):
        if isinstance(self._runner, Runner):
            return self._runner._run_batch(**kwargs)

        if isinstance(self._runner, SimpleRunner):
            results = []
            params = Params(args, kwargs)
            for iparams in params.iter(batch_to_single_list):
                results.append(self._runner._run(*iparams.args, **iparams.kwargs))
            return single_list_to_batch(results)
