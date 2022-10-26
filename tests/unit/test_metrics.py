from __future__ import annotations

import bentoml


def test_metrics_initialization():
    o = bentoml.metrics.Gauge(name="test_metrics", documentation="test")
    assert isinstance(o, bentoml.metrics._LazyMetric)
    assert o._proxy is None
    o = bentoml.metrics.Histogram(name="test_metrics", documentation="test")
    assert isinstance(o, bentoml.metrics._LazyMetric)
    assert o._proxy is None
    o = bentoml.metrics.Counter(name="test_metrics", documentation="test")
    assert isinstance(o, bentoml.metrics._LazyMetric)
    assert o._proxy is None
    o = bentoml.metrics.Summary(name="test_metrics", documentation="test")
    assert isinstance(o, bentoml.metrics._LazyMetric)
    assert o._proxy is None
