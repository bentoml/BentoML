from __future__ import annotations

import os
import sys
import typing as t
import logging
from typing import TYPE_CHECKING
from functools import partial

if TYPE_CHECKING:
    from prometheus_client.metrics_core import Metric

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
        """
        Starts a WSGI server for prometheus metrics as a daemon thread.

        Args:
            port: Port to listen on.
            addr: Address to listen on.
        """
        self.prometheus_client.start_http_server(
            port=port,
            addr=addr,
            registry=self.registry,
        )

    def make_wsgi_app(self) -> ext.WSGIApp:
        """
        Create a WSGI app which serves the metrics from a registry.

        Returns:
            WSGIApp: A WSGI app which serves the metrics from a registry.
        """
        return self.prometheus_client.make_wsgi_app(registry=self.registry)  # type: ignore (unfinished prometheus types)

    def generate_latest(self):
        """
        Returns metrics from the registry in latest text format as a string.

        This function ensures that multiprocess is setup correctly.

        Returns:
            str: Metrics in latest text format. Refer to `Exposition format <https://prometheus.io/docs/instrumenting/exposition_formats/#exposition-formats>`_ for details.
        """
        if self.multiproc:
            registry = self.prometheus_client.CollectorRegistry()
            self.prometheus_client.multiprocess.MultiProcessCollector(registry)
            return self.prometheus_client.generate_latest(registry)
        else:
            return self.prometheus_client.generate_latest()

    def text_string_to_metric_families(self) -> t.Generator[Metric, None, None]:
        """
        Parse Prometheus text format from a unicode string.

        Returns:
            Generator[Metric, None, None]: A generator of `Metric <https://prometheus.io/docs/concepts/metric_types/>`_ objects.
        """
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
        o = partial(self.prometheus_client.Histogram, registry=self.registry)
        o.__doc__ = """
        A Histogram tracks the size and number of events in a given bucket.

        Histograms are often used to aggregatable calculation of quantiles.
        Some notable examples include measuring response latency, request size.

        A quick example of a Histogram:

        .. code-block:: python

           from bentoml.metrics import Histogram

           h = Histogram('request_size_bytes', 'Request size (bytes)')

           @svc.api(input=JSON(), output=JSON())
           def predict(input_data: dict[str, str]):
               h.observe(512)  # Observe 512 (bytes)
               ...

        ``observe()`` will observe for given amount of time.
        Usually, this value are positive or zero. Negative values are accepted but will
        prevent current versions of Prometheus from properly detecting counter resets in the `sum of observations <https://prometheus.io/docs/practices/histograms/#count-and-sum-of-observations>`_.

        Histograms also provide ``time()``, which times a block of code or function, and observe for a given duration amount.
        This function can also be used as a context manager.

        .. tab-set::

           .. tab-item:: Example

              .. code-block:: python

                 from bentoml.metrics import Histogram

                 REQUEST_TIME = Histogram('response_latency_seconds', 'Response latency (seconds)')

                 @REQUEST_TIME.time()
                 def create_response(request):
                     body = await request.json()
                     return Response(body)

           .. tab-item:: Context Manager

              .. code-block:: python

                 from bentoml.metrics import Histogram

                 REQUEST_TIME = Histogram('response_latency_seconds', 'Response latency (seconds)')

                 def create_response(request):
                     body = await request.json()
                     with REQUEST_TIME.time():
                         ...

        The default buckets are intended to cover a typical web/rpc request from milliseconds to seconds.
        See :ref:`configuration guides <guides/configuration:Configuration>` to see how to customize the buckets.

        Args:
            name (str): The name of the metric.
            documentation (str): A documentation string.
            labelnames (tuple[str]): A tuple of strings specifying the label names for the metric. Defaults to ``()``.
            namespace (str): The namespace of the metric. Defaults to an empty string.
            subsystem (str): The subsystem of the metric. Defaults to an empty string.
            unit (str): The unit of the metric. Defaults to an empty string.
            buckets (list[float]): A list of float representing a bucket. Defaults to ``(.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0, 7.5, 10.0, INF)``.
        """
        return o

    @property
    def Counter(self):
        o = partial(self.prometheus_client.Counter, registry=self.registry)
        o.__doc__ = """
        A Counter tracks counts of events or running totals.

        .. epigraph::

           It is a cumulative metric that represents a single `monotonically increasing counter <https://prometheus.io/docs/concepts/metric_types/#counter>`_ whose value can only increase or be reset to zero on restart.

        Some notable examples include counting the number of requests served, tasks completed, or errors.

        If you need to go down, uses :func:`bentoml.metrics.Gauge` instead.

        A quick example of a Counter:

        .. code-block:: python

           from bentoml.metrics import Counter

           c = Counter('failures', 'Total number of failures requests')

           @svc.api(input=JSON(), output=JSON())
           def predict(input_data: dict[str, str]):
               if input_data['fail']:
                   c.inc()  # increment by 1 by default

        ``inc()`` can optionally pass in a ``exemplar``, which is a dictionary of keys and values, defined :github:`here <OpenObservability/OpenMetrics/blob/main/specification/OpenMetrics.md#exemplars>`.

        ``inc()`` can also increment by any given amount:

        .. code-block:: python

           c.inc(2.1)

        ``count_exceptions()`` can be used as both a decorator and context manager to count exceptions raised.

        .. tab-set::

           .. tab-item:: Decorator

              .. code-block:: python

                 from bentoml.metrics import Counter

                 c = Counter('failures', 'Total number of failures requests')

                 @c.count_exceptions()
                 @svc.api(input=JSON(), output=JSON())
                 def predict(input_data: dict[str, str]):
                     if input_data['acc'] < 0.5:
                         raise ValueError("Given data is not accurate.")

           .. tab-item:: Context Manager

              .. code-block:: python

                 from bentoml.metrics import Histogram

                 c = Counter('failures', 'Total number of failures requests')

                 @svc.api(input=JSON(), output=JSON())
                 def predict(input_data: dict[str, str]):
                     with c.count_exceptions():
                         if input_data['acc'] < 0.5:
                             raise ValueError("Given data is not accurate.")
                     with c.count_exceptions(RuntimeError):
                         if input_data['output'] is None:
                             raise RuntimeError("Given pre-processing logic is invalid")

        ``count_exceptions()`` will optionally take in an exception to only track specific exceptions.

        .. code-block:: python

           ...
           with c.count_exceptions(RuntimeError):
               if input_data['output'] is None:
                   raise RuntimeError("Given pre-processing logic is invalid")

        Args:
            name (str): The name of the metric.
            documentation (str): A documentation string.
            labelnames (tuple[str]): A tuple of strings specifying the label names for the metric. Defaults to ``()``.
            namespace (str): The namespace of the metric. Defaults to an empty string.
            subsystem (str): The subsystem of the metric. Defaults to an empty string.
            unit (str): The unit of the metric. Defaults to an empty string.
        """
        return o

    @property
    def Summary(self):
        o = partial(self.prometheus_client.Summary, registry=self.registry)
        o.__doc__ = """

        A Summary tracks the size and `samples observations (usually things like request durations and response sizes).`.

        While it also provides a total count of observations and a sum of all observed values,
        it calculates configurable quantiles over a sliding time window.

        Notable examples include request latency and response size.

        A quick example of a Summary:

        .. code-block:: python

           from bentoml.metrics import Summary

           s = Summary('request_size_bytes', 'Request size (bytes)')

           @svc.api(input=JSON(), output=JSON())
           def predict(input_data: dict[str, str]):
               s.observe(512)  # Observe 512 (bytes)
               ...

        ``observe()`` will observe for given amount of time.
        Usually, this value are positive or zero. Negative values are accepted but will
        prevent current versions of Prometheus from properly detecting counter resets in the `sum of observations <https://prometheus.io/docs/practices/histograms/#count-and-sum-of-observations>`_.

        Similar to :meth:`bentoml.metrics.Histogram`, ``time()`` can also be used as a decorator or context manager.

        .. tab-set::

           .. tab-item:: Example

              .. code-block:: python

                 from bentoml.metrics import Histogram

                 s = Summary('response_latency_seconds', 'Response latency (seconds)')

                 @s.time()
                 def create_response(request):
                     body = await request.json()
                     return Response(body)

           .. tab-item:: Context Manager

              .. code-block:: python

                 from bentoml.metrics import Histogram

                 s = Summary('response_latency_seconds', 'Response latency (seconds)')

                 def create_response(request):
                     body = await request.json()
                     with s.time():
                         ...

        Args:
            name (str): The name of the metric.
            documentation (str): A documentation string.
            labelnames (tuple[str]): A tuple of strings specifying the label names for the metric. Defaults to ``()``.
            namespace (str): The namespace of the metric. Defaults to an empty string.
            subsystem (str): The subsystem of the metric. Defaults to an empty string.
            unit (str): The unit of the metric. Defaults to an empty string.
        """
        return o

    @property
    def Gauge(self):
        o = partial(self.prometheus_client.Gauge, registry=self.registry)
        o.__doc__ = """
        A Gauge represents a single numerical value that can arbitrarily go up and down.

        Gauges are typically used to for report instantaneous values like temperatures or current memory usage.
        One can think of Gauge as a :meth:`bentoml.metrics.Counter` that can go up and down.

        Notable examples include in-progress requests, number of item in a queue, and free memory.

        A quick example of a Gauge:

        .. code-block:: python

           from bentoml.metrics import Gauge

           g = Gauge('inprogress_request', 'Request inprogress')

           @svc.api(input=JSON(), output=JSON())
           def predict(input_data: dict[str, str]):
               g.inc()  # increment by 1 by default
               g.dec(10) # decrement by any given value
               g.set(0)  # set to a given value
               ...

        .. note::

           By default, ``inc()`` and ``dec()`` will increment and decrement by 1 respectively.

        Gauge also provide ``track_inprogress()``, to track inprogress object.
        This function can also be used as either a context manager or a decorator.

        .. tab-set::

           .. tab-item:: Example

              .. code-block:: python

                 from bentoml.metrics import Gauge

                 g = Gauge('inprogress_request', 'Request inprogress')

                 @svc.api(input=JSON(), output=JSON())
                 @g.track_inprogress()
                 def predict(input_data: dict[str, str]):
                     ...

           .. tab-item:: Context Manager

              .. code-block:: python

                 from bentoml.metrics import Gauge

                 g = Gauge('inprogress_request', 'Request inprogress')

                 @svc.api(input=JSON(), output=JSON())
                 def predict(input_data: dict[str, str]):
                     with g.track_inprogress():
                         ...

                The gauge will increment when the context is entered and decrement when the context is exited.

        Args:
            name (str): The name of the metric.
            documentation (str): A documentation string.
            labelnames (tuple[str]): A tuple of strings specifying the label names for the metric. Defaults to ``()``.
            namespace (str): The namespace of the metric. Defaults to an empty string.
            subsystem (str): The subsystem of the metric. Defaults to an empty string.
            unit (str): The unit of the metric. Defaults to an empty string.
            multiprocess_mode (str): The multiprocess mode of the metric. Defaults to ``all``. Available options
                                     are (``all``, ``min``, ``max``, ``livesum``, ``liveall``)
        """
        return o

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
