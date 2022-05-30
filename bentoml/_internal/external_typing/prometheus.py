from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import threading

    from prometheus_client.samples import Sample
    from prometheus_client.samples import Exemplar
    from prometheus_client.samples import Timestamp
    from prometheus_client.registry import RestrictedRegistry
    from prometheus_client.gc_collector import GCCollector
    from prometheus_client.process_collector import ProcessCollector
    from prometheus_client.platform_collector import PlatformCollector

    Collector = PlatformCollector | GCCollector | ProcessCollector

    class Metric:
        """A single metric family and its samples.

        This is intended only for internal use by the instrumentation client.

        Custom collectors should use GaugeMetricFamily, CounterMetricFamily
        and SummaryMetricFamily instead.
        """

        name: str
        documentation: str
        type: str
        samples: list[Sample]

        def __init__(self, name: str, documentation: str, typ: str, unit: str = ...):
            ...

        def add_sample(
            self,
            name: str,
            labels: dict[str, str],
            value: float,
            timestamp: float | Timestamp | None = ...,
            exemplar: Exemplar | None = ...,
        ) -> None:
            ...

        def __eq__(self, o: Metric) -> bool:
            ...

        def __repr__(self) -> str:
            ...

        def _restricted_metric(self, names: list[str]) -> Metric | None:
            ...

    class CollectorRegistry:
        """Metric collector registry.

        Collectors must have a no-argument method 'collect' that returns a list of
        Metric objects. The returned metrics should be consistent with the Prometheus
        exposition formats.
        """

        _collector_to_name: dict[t.Type[Collector], str]
        _names_to_collectors: dict[str, t.Type[Collector]]
        _auto_describe: bool
        _lock: threading.Lock

        def __init__(
            self, auto_describe: bool = ..., target_info: dict[str, t.Any] | None = ...
        ):
            ...

        def register(self, collector: Collector) -> None:
            ...

        def unregister(self, collector: Collector) -> None:
            ...

        def _get_names(self, collector: Collector) -> list[str]:
            ...

        def collect(self) -> t.Generator[Metric, None, None]:
            ...

        def restricted_registry(self, names: list[str]) -> RestrictedRegistry:
            ...

        def set_target_info(self, labels: dict[str, str]) -> None:
            ...

        def get_target_info(self) -> dict[str, str]:
            ...

        def _target_info_metric(self) -> Metric:
            ...

        def get_sample_value(self, name: str, labels: dict[str, str]) -> float | None:
            ...

    __all__ = ["Metric", "CollectorRegistry"]
