import typing as t

import attr

_T = t.TypeVar("_T")


@attr.s(slots=False)
class Runner:
    _model_path = attr.ib(default="", type=str)

    CPU = attr.ib(default=1.0, type=float)
    RAM = attr.ib(default="100M", type=str)

    GPU = attr.ib(default=0.0, type=float)
    _on_gpu = attr.ib(default=False, type=bool)

    dynamic_batching = attr.ib(default=True, type=bool)
    max_batch_size = attr.ib(default=10000, type=int)
    max_latency_ms = attr.ib(default=10000, type=int)

    @property
    def num_concurrency(self):
        raise NotImplementedError

    @property
    def num_replica(self):
        raise NotImplementedError

    def _setup(self, *args, **kwargs):
        raise NotImplementedError

    def _run_batch(self, input_data: "_T") -> "_T":
        raise NotImplementedError
