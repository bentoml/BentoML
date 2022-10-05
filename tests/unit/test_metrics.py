from __future__ import annotations

import sys
import typing as t

import pytest

import bentoml


@pytest.mark.parametrize(
    "metrics_type",
    filter(
        lambda x: isinstance(x, bentoml.metrics.metrics.MetricWrapperBase),
        dir(bentoml.metrics),
    ),
)
def test_metrics_initialization(
    metrics_type: t.Type[bentoml.metrics.metrics.MetricWrapperBase],
):
    _ = metrics_type(name="test_metrics", documentation="test")
    assert "prometheus_client" not in sys.modules
