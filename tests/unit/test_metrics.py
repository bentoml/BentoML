from __future__ import annotations

from typing import TYPE_CHECKING

import bentoml

if TYPE_CHECKING:
    from bentoml._internal.server.metrics.prometheus import PrometheusClient


def test_metrics_initialization():
    o = bentoml.metrics.Gauge(name="test_metrics", documentation="test")
    assert isinstance(o, bentoml.metrics._LazyMetric)
    o = bentoml.metrics.Histogram(name="test_metrics", documentation="test")
    assert isinstance(o, bentoml.metrics._LazyMetric)
    o = bentoml.metrics.Counter(name="test_metrics", documentation="test")
    assert isinstance(o, bentoml.metrics._LazyMetric)
    o = bentoml.metrics.Summary(name="test_metrics", documentation="test")
    assert isinstance(o, bentoml.metrics._LazyMetric)


def test_metrics_type(prom_client: PrometheusClient):
    o = bentoml.metrics.Counter(name="test_metrics", documentation="test")
    assert o._attr == "Counter"
    assert o._proxy is None
    o.inc()
    assert isinstance(o._proxy, prom_client.prometheus_client.Counter)
