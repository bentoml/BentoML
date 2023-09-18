from __future__ import annotations

import os
import sys
import typing as t
import logging
from typing import TYPE_CHECKING
from functools import partial

if TYPE_CHECKING:
    from ... import external_typing as ext

logger = logging.getLogger(__name__)


class PrometheusClient:
    def __init__(
        self,
        *,
        multiproc: bool = True,
        multiproc_dir: str | None = None,
    ):
        """
        PrometheusClient is BentoML's own prometheus client that extends the official Python client.

        It sets up a multiprocess dir for Prometheus to work in multiprocess mode, which is required
        for BentoML to work in production.

        .. note::

           For Prometheus to behave properly, ``prometheus_client`` must be imported after this client
           is called. This has to do with ``prometheus_client`` relies on ``PROMEHEUS_MULTIPROC_DIR``, which
           will be set by this client.

        For API documentation, refer to https://docs.bentoml.org/en/latest/reference/metrics.html.
        """
        if multiproc:
            assert multiproc_dir is not None, "multiproc_dir must be provided"

        self.multiproc = multiproc
        self.multiproc_dir: str | None = multiproc_dir
        self._registry = None
        self._imported = False
        self._pid: int | None = None

    @property
    def prometheus_client(self):
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
        import prometheus_client.parser
        import prometheus_client.metrics
        import prometheus_client.exposition
        import prometheus_client.metrics_core
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
        self.prometheus_client.start_http_server(
            port=port,
            addr=addr,
            registry=self.registry,
        )

    start_wsgi_server = start_http_server

    def write_to_textfile(self, path: str) -> None:
        """
        Write metrics to given path. This is intended to be used with
        the Node expoerter textfile collector.

        Args:
            path: path to write the metrics to. This file must end
                with '.prom' for the textfile collector to process it.
        """
        self.prometheus_client.write_to_textfile(path, registry=self.registry)

    def make_wsgi_app(self) -> ext.WSGIApp:
        # Used by gRPC prometheus server.
        return self.prometheus_client.make_wsgi_app(registry=self.registry)  # type: ignore (unfinished prometheus types)

    def generate_latest(self):
        if self.multiproc:
            registry = self.prometheus_client.CollectorRegistry()
            self.prometheus_client.multiprocess.MultiProcessCollector(registry)
            return self.prometheus_client.generate_latest(registry)
        else:
            return self.prometheus_client.generate_latest()

    def text_string_to_metric_families(self) -> t.Generator[Metric, None, None]:
        yield from self.prometheus_client.parser.text_string_to_metric_families(
            self.generate_latest().decode("utf-8")
        )

    @property
    def CONTENT_TYPE_LATEST(self) -> str:
        """
        Returns:
            str: Content type of the latest text format
        """
        return self.prometheus_client.CONTENT_TYPE_LATEST

    # For all of the documentation for instruments metrics below, we will extend
    # upon prometheus_client's documentation, since their code segment aren't rst friendly, and
    # not that easy to read.

    @property
    def Histogram(self):
        return partial(self.prometheus_client.Histogram, registry=self.registry)

    @property
    def Counter(self):
        return partial(self.prometheus_client.Counter, registry=self.registry)

    @property
    def Summary(self):
        return partial(self.prometheus_client.Summary, registry=self.registry)

    @property
    def Gauge(self):
        return partial(self.prometheus_client.Gauge, registry=self.registry)

    @property
    def Info(self):
        raise RuntimeError("Info is not supported in Prometheus multiprocess mode.")

    @property
    def Enum(self):
        raise RuntimeError("Enum is not supported in Prometheus multiprocess mode.")

    @property
    def Metric(self):
        """
        A Metric family and its samples.

        This is a base class to be used by instrumentation client. Custom collectors should use ``bentoml.metrics.metrics_core.GaugeMetricFamily``, ``bentoml.metrics.metrics_core.CounterMetricFamily``, ``bentoml.metrics.metrics_core.SummaryMetricFamily`` instead.
        """
        return partial(self.prometheus_client.Metric, registry=self.registry)
