from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from ._internal.configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ._internal.server.metrics.prometheus import PrometheusClient

logger = logging.getLogger(__name__)

# NOTE: We have to set our docstring here due to the fact that
# we are lazy loading the metrics. This means that the docstring
# won't be discovered until the metrics is initialized.
# this won't work with help() or doocstring on Sphinx.
# While this is less than optimal, we will do this since 'bentoml.metrics'
# is a public API.
_MAKE_WSGI_APP_DOCSTRING = """\
Create a WSGI app which serves the metrics from a registry.

Returns:
    WSGIApp: A WSGI app which serves the metrics from a registry.
"""
_GENERATE_LATEST_DOCSTRING = """\
Returns metrics from the registry in latest text format as a string.

This function ensures that multiprocess is setup correctly.

Returns:
    str: Metrics in latest text format. Refer to `Exposition format <https://prometheus.io/docs/instrumenting/exposition_formats/#exposition-formats>`_ for details.
"""
_TEXT_STRING_TO_METRIC_DOCSTRING = """
Parse Prometheus text format from a unicode string.

Returns:
    Metric: A generator that yields `Metric <https://prometheus.io/docs/concepts/metric_types/>`_ objects.
"""
_HISTOGRAM_DOCSTRING = """\
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
_COUNTER_DOCSTRING = """
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
_SUMMARY_DOCSTRING = """
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
_GAUGE_DOCSTRING = """
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

# This sets of functions are implemented in the PrometheusClient class
_INTERNAL_FN_IMPL = {
    "make_wsgi_app": _MAKE_WSGI_APP_DOCSTRING,
    "generate_latest": _GENERATE_LATEST_DOCSTRING,
    "text_string_to_metric_families": _TEXT_STRING_TO_METRIC_DOCSTRING,
}
_NOT_IMPLEMENTED = [
    "delete_from_gateway",
    "instance_ip_grouping_key",
    "push_to_gateway",
    "pushadd_to_gateway",
]
_NOT_SUPPORTED = [
    "GC_COLLECTOR",
    "GCCollector",
    "PLATFORM_COLLECTOR",
    "PlatformCollector",
    "PROCESS_COLLECTOR",
    "ProcessCollector",
    "REGISTRY",
    "CONTENT_TYPE_LATEST",
    "start_http_server",
    "start_wsgi_server",
    "make_asgi_app",
    "write_to_textfile",
] + _NOT_IMPLEMENTED
_docstring = {
    "Counter": _COUNTER_DOCSTRING,
    "Histogram": _HISTOGRAM_DOCSTRING,
    "Summary": _SUMMARY_DOCSTRING,
    "Gauge": _GAUGE_DOCSTRING,
}
_docstring.update(_INTERNAL_FN_IMPL)


def __dir__() -> list[str]:
    # This is for IPython and IDE autocompletion.
    metrics_client = BentoMLContainer.metrics_client.get()
    return list(set(dir(metrics_client.prometheus_client)) - set(_NOT_SUPPORTED))


def __getattr__(item: t.Any):
    if item in _NOT_SUPPORTED:
        raise NotImplementedError(
            f"{item} is not supported when using '{__name__}'. See https://docs.bentoml.org/en/latest/reference/metrics.html."
        )
    # This is the entrypoint for all bentoml.metrics.*
    return _LazyMetric(item, docstring=_docstring.get(item))


class _LazyMetric:
    __slots__ = ("_attr", "_proxy", "_initialized", "_args", "_kwargs", "__doc__")

    def __init__(self, attr: str, docstring: str | None = None):
        self._attr = attr
        self.__doc__ = docstring
        self._proxy = None
        self._initialized = False
        self._args: tuple[t.Any, ...] = ()
        self._kwargs: dict[str, t.Any] = {}

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """
        Lazily initialize the metrics object.

        Args:
            *args: Arguments to pass to the metrics object.
            **kwargs: Keyword arguments to pass to the metrics object.
        """
        if "registry" in kwargs:
            raise ValueError(
                f"'registry' should not be passed when using '{__name__}.{self._attr}'. See https://docs.bentoml.org/en/latest/reference/metrics.html."
            )
        self._args = args
        self._kwargs = kwargs
        if self._attr in _INTERNAL_FN_IMPL:
            # first-class function implementation from BentoML Prometheus client.
            # In this case, the function will be called directly.
            return self._load_proxy()
        return self

    def __getattr__(self, item: t.Any) -> t.Any:
        if item in self.__slots__:
            raise AttributeError(f"Attribute {item} is private to {self}.")
        if self._proxy is None:
            self._proxy = self._load_proxy()
        assert self._initialized and self._proxy is not None
        return getattr(self._proxy, item)

    def __dir__(self) -> list[str]:
        if self._proxy is None:
            self._proxy = self._load_proxy()
        assert self._initialized and self._proxy is not None
        return dir(self._proxy)

    @inject
    def _load_proxy(
        self,
        metrics_client: PrometheusClient = Provide[BentoMLContainer.metrics_client],
    ) -> None:
        client_impl = (
            metrics_client
            if self._attr in dir(metrics_client)
            else metrics_client.prometheus_client
        )
        proxy = getattr(client_impl, self._attr)(*self._args, **self._kwargs)
        self._initialized = True
        return proxy
