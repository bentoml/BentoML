from __future__ import annotations

import os
import re
import sys
import typing as t
import logging
from typing import TYPE_CHECKING
from functools import partial

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from types import ModuleType

    from ...external_typing import prometheus as ext


# filter out unnecessary metrics
EXCLUDE_PATHS = [
    "/metrics",
    "/static_content",
    "/static_content/*",
    "/docs.json",
    "/openapi.json",
    "/favicon.ico",
]


class PrometheusClient:
    def __init__(
        self,
        *,
        namespace: str = "",
        multiproc: bool = True,
        multiproc_dir: str | None = None,
    ):
        """
        Set up multiproc_dir for prometheus to work in multiprocess mode,
        which is required when working with Gunicorn server

        Warning: for this to work, prometheus_client library must be imported after
        this function is called. It relies on the os.environ['PROMETHEUS_MULTIPROC_DIR']
        to properly setup for multiprocess mode
        """
        if multiproc:
            assert multiproc_dir is not None, "multiproc_dir must be provided"

        self.multiproc = multiproc
        self.namespace = namespace
        self.multiproc_dir: str | None = multiproc_dir
        self._registry: ext.CollectorRegistry | None = None
        self._imported = False
        self._pid: int | None = None
        self._exc_paths: list[str] = EXCLUDE_PATHS

    @property
    def prometheus_client(self) -> ModuleType:
        if self.multiproc and not self._imported:
            # step 1: check environment
            assert (
                "prometheus_client" not in sys.modules
            ), "prometheus_client is already imported, multiprocessing will not work properly"

            assert (
                self.multiproc_dir
            ), f"Invalid prometheus multiproc directory: {self.multiproc_dir}"
            assert os.path.isdir(self.multiproc_dir)

            os.environ["PROMETHEUS_MULTIPROC_DIR"] = self.multiproc_dir

        # step 2:
        import prometheus_client
        import prometheus_client.multiprocess

        self._imported = True
        return prometheus_client

    @property
    def registry(self):
        if self._registry is None:
            if self.multiproc:
                self._pid = os.getpid()
            self._registry = self.prometheus_client.REGISTRY
        else:
            if self.multiproc:
                assert self._pid is not None
                assert (
                    os.getpid() == self._pid
                ), "The current process's different than the process which the prometheus client gets created"

        return self._registry

    def __del__(self):
        self.mark_process_dead()

    def mark_process_dead(self) -> None:
        if self.multiproc:
            assert self._pid is not None
            assert (
                os.getpid() == self._pid
            ), "The current process's different than the process which the prometheus client gets created"
            self.prometheus_client.multiprocess.mark_process_dead(self._pid)

    def start_http_server(self, port: int, addr: str = "") -> None:
        self.prometheus_client.start_http_server(  # type: ignore
            port=port,
            addr=addr,
            registry=self.registry,
        )

    def generate_latest(self) -> bytes:
        if self.multiproc:
            registry = self.prometheus_client.CollectorRegistry()
            self.prometheus_client.multiprocess.MultiProcessCollector(registry)
            return t.cast(bytes, self.prometheus_client.generate_latest(registry))
        else:
            return self.prometheus_client.generate_latest()

    def get_metrics_report(
        self, filter_: list[str] | None = None
    ) -> list[dict[str, str | float]] | None:
        metrics_text = self.generate_latest().decode("utf-8")

        if not metrics_text:
            return []

        if filter_ is not None:
            filter_ = self._exc_paths + filter_
        else:
            filter_ = self._exc_paths

        from prometheus_client.parser import (
            text_string_to_metric_families,  # type: ignore
        )

        metric: ext.Metric

        def truncate(metrics: dict[str, str | float]) -> bool:
            for m in filter_:
                if re.match(m, str(metrics["endpoint"])) is not None:
                    return False
            return True

        report_: list[dict[str, str | float]] = []

        for metric in text_string_to_metric_families(metrics_text):
            if (
                metric.type == "counter"
                and metric.name.startswith("BENTOML_")
                and metric.name.endswith("_request")
            ):
                metric_ = [
                    {**sample.labels, "value": sample.value}
                    for sample in metric.samples
                ]
                report_ += list(filter(truncate, metric_))
        return report_

    @property
    def CONTENT_TYPE_LATEST(self) -> str:
        return self.prometheus_client.CONTENT_TYPE_LATEST

    @property
    def Histogram(self):
        return partial(
            self.prometheus_client.Histogram,
            namespace=self.namespace,
            registry=self.registry,
        )

    @property
    def Counter(self):
        return partial(
            self.prometheus_client.Counter,
            namespace=self.namespace,
            registry=self.registry,
        )

    @property
    def Summary(self):
        return partial(
            self.prometheus_client.Summary,
            namespace=self.namespace,
            registry=self.registry,
        )

    @property
    def Gauge(self):
        return partial(
            self.prometheus_client.Gauge,
            namespace=self.namespace,
            registry=self.registry,
        )
