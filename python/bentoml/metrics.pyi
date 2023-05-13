from __future__ import annotations

from typing import Generator

import prometheus_client

from bentoml._internal import external_typing as ext

CONTENT_TYPE_LATEST = prometheus_client.CONTENT_TYPE_LATEST
Counter = prometheus_client.Counter
Histogram = prometheus_client.Histogram
Summary = prometheus_client.Summary
Gauge = prometheus_client.Gauge
Info = prometheus_client.Info
Enum = prometheus_client.Enum
Metric = prometheus_client.Metric

def start_http_server(port: int, addr: str = ...) -> None: ...
def make_wsgi_app() -> ext.WSGIApp: ...
def generate_latest() -> bytes: ...
def text_string_to_metric_families() -> Generator[Metric, None, None]: ...
