from __future__ import annotations

import logging
import os
import re
import typing as t
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prometheus_client import Metric

    from ... import external_typing as ext

logger = logging.getLogger(__name__)


class PrometheusClient:
    def __init__(self, *, multiproc: bool = True):
        """
        PrometheusClient is BentoML's own prometheus client that extends the official Python client.

        It sets up a multiprocess dir for Prometheus to work in multiprocess mode, which is required
        for BentoML to work in production.

        .. note::

           For Prometheus to behave properly, ``prometheus_client`` must be imported after this client
           is called. This has to do with ``prometheus_client`` relies on ``PROMEHEUS_MULTIPROC_DIR``, which
           will be set by this client.

        For API documentation, refer to https://docs.bentoml.com/en/latest/reference/metrics.html.
        """
        self.multiproc = multiproc
        self._registry = None
        self._imported = False
        self._pid: int | None = None

    @property
    def prometheus_client(self):
        import prometheus_client
        import prometheus_client.exposition
        import prometheus_client.metrics
        import prometheus_client.metrics_core
        import prometheus_client.multiprocess
        import prometheus_client.parser

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
                assert os.getpid() == self._pid, (
                    "The current process's different than the process which the prometheus client gets created"
                )

        return self._registry

    def __del__(self):
        self.mark_process_dead()

    def mark_process_dead(self) -> None:
        if self.multiproc:
            assert self._pid is not None
            assert os.getpid() == self._pid, (
                "The current process's different than the process which the prometheus client gets created"
            )
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
            raw_output = self.prometheus_client.generate_latest(registry)
            return self._fix_histogram_ordering(raw_output)
        else:
            raw_output = self.prometheus_client.generate_latest()
            return self._fix_histogram_ordering(raw_output)

    def _fix_histogram_ordering(self, prometheus_output: bytes) -> bytes:
        """
        Fix histogram metric ordering to comply with Prometheus text format specification.

        The Prometheus format requires histogram metrics to be grouped by metric name with:
        1. All _bucket metrics for a histogram (in ascending order of 'le' values)
        2. Followed by _count metric
        3. Followed by _sum metric

        Args:
            prometheus_output: Raw Prometheus format output

        Returns:
            Properly ordered Prometheus format output
        """
        lines = prometheus_output.decode("utf-8").strip().split("\n")

        # Separate comments/help lines from metric lines
        comment_lines = []
        metric_lines = []

        for line in lines:
            if line.startswith("#") or line.strip() == "":
                comment_lines.append(line)
            else:
                metric_lines.append(line)

        # Group metrics by base name (without _bucket, _count, _sum suffixes)
        metrics_by_base = {}
        non_histogram_metrics = []

        for line in metric_lines:
            if not line.strip():
                continue

            # Extract metric name (everything before the first space or '{')
            if "{" in line:
                metric_name = line.split("{")[0]
            else:
                metric_name = line.split(" ")[0]

            # Check if this is a histogram metric
            if metric_name.endswith("_bucket"):
                base_name = metric_name[:-7]  # Remove '_bucket'
                if base_name not in metrics_by_base:
                    metrics_by_base[base_name] = {"bucket": [], "count": [], "sum": []}
                metrics_by_base[base_name]["bucket"].append(line)
            elif metric_name.endswith("_count"):
                base_name = metric_name[:-6]  # Remove '_count'
                if base_name not in metrics_by_base:
                    metrics_by_base[base_name] = {"bucket": [], "count": [], "sum": []}
                metrics_by_base[base_name]["count"].append(line)
            elif metric_name.endswith("_sum"):
                base_name = metric_name[:-4]  # Remove '_sum'
                if base_name not in metrics_by_base:
                    metrics_by_base[base_name] = {"bucket": [], "count": [], "sum": []}
                metrics_by_base[base_name]["sum"].append(line)
            else:
                non_histogram_metrics.append(line)

        # Function to extract 'le' value for bucket sorting
        def extract_le_value(bucket_line: str) -> float:
            try:
                # Find le="value" in the line
                match = re.search(r'le="([^"]+)"', bucket_line)
                if match:
                    le_val = match.group(1)
                    if le_val == "+Inf":
                        return float("inf")
                    return float(le_val)
                return float("inf")  # Default if parsing fails
            except:
                return float("inf")

        # Rebuild the output with proper ordering
        result_lines = comment_lines.copy()

        # Add non-histogram metrics first
        result_lines.extend(non_histogram_metrics)

        # Add histogram metrics in proper order
        for base_name in sorted(metrics_by_base.keys()):
            hist_data = metrics_by_base[base_name]

            # Sort buckets by 'le' value in ascending order
            sorted_buckets = sorted(hist_data["bucket"], key=extract_le_value)
            result_lines.extend(sorted_buckets)

            # Add count metrics
            result_lines.extend(hist_data["count"])

            # Add sum metrics
            result_lines.extend(hist_data["sum"])

        return "\n".join(result_lines).encode("utf-8")

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

        This is a base class to be used by instrumentation client.
        Custom collectors should use
        ``prometheus_client.metrics_core.GaugeMetricFamily``,
        ``prometheus_client.metrics_core.CounterMetricFamily``,
        ``prometheus_client.metrics_core.SummaryMetricFamily`` instead.
        """
        return partial(self.prometheus_client.Metric, registry=self.registry)
