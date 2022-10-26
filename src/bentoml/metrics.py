from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

if TYPE_CHECKING:
    from ._internal.server.metrics.prometheus import PrometheusClient

logger = logging.getLogger(__name__)

from ._internal.configuration.containers import BentoMLContainer


def __dir__() -> list[str]:
    metrics_client = BentoMLContainer.metrics_client.get()
    return dir(metrics_client.prometheus_client)


class LazyObject:
    def __init__(self, attr: t.Any):
        self.__bentoml_metrics_attr__ = attr
        self._proxy = None
        self._initialized = False
        self._args: tuple[t.Any, ...] = ()
        self._kwargs: dict[str, t.Any] = {}

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        # This is where we lazy load the proxy object.
        self._args = args
        self._kwargs = kwargs
        return self

    def __getattr__(self, item: t.Any) -> t.Any:
        if self._proxy is None:
            self._load_proxy()
        assert self._initialized
        return getattr(self._proxy, item)

    @inject
    def _load_proxy(
        self,
        metrics_client: PrometheusClient = Provide[BentoMLContainer.metrics_client],
    ) -> None:
        parent = (
            metrics_client
            if self.__bentoml_metrics_attr__ in dir(metrics_client)
            else metrics_client.prometheus_client
        )
        self._proxy = getattr(parent, self.__bentoml_metrics_attr__)(
            *self._args, **self._kwargs
        )
        self._initialized = True


def __getattr__(item: t.Any):
    return LazyObject(item)
