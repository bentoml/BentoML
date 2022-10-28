from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from ._internal.utils import warn_experimental
from ._internal.utils import add_experimental_docstring

if TYPE_CHECKING:
    from ._internal.server.metrics.prometheus import PrometheusClient

logger = logging.getLogger(__name__)

from ._internal.configuration.containers import BentoMLContainer

# This sets of functions are implemented in the PrometheusClient class
_INTERNAL_IMPL = [
    "start_http_server",
    "start_wsgi_server",
    "make_wsgi_app",
    "make_asgi_app",
    "generate_latest",
    "text_string_to_metric_families",
    "write_to_textfile",
]
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
] + _NOT_IMPLEMENTED
_PROPERTY = ["CONTENT_TYPE_LATEST"]


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
    if item in _PROPERTY:
        logger.warning(
            "'%s' is a '@property', which means there is no lazy loading. See https://docs.bentoml.org/en/latest/reference/metrics.html.",
            item,
        )
        return getattr(_LazyMetric(item), item)
    return _LazyMetric(item)


class _LazyMetric:
    __slots__ = ("_attr", "_proxy", "_initialized", "_args", "_kwargs")

    def __init__(self, attr: str):
        self._attr = attr
        self._proxy = None
        self._initialized = False
        self._args: tuple[t.Any, ...] = ()
        self._kwargs: dict[str, t.Any] = {}

    @add_experimental_docstring
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
        warn_experimental("%s.%s" % (__name__, self._attr))
        self._args = args
        self._kwargs = kwargs
        if self._attr in _INTERNAL_IMPL:
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
        if self._attr in _PROPERTY:
            return self._proxy
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
        proxy = getattr(client_impl, self._attr)
        if callable(proxy):
            proxy = proxy(*self._args, **self._kwargs)
        self._initialized = True
        return proxy
