from __future__ import annotations

import typing as t
import logging
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

logger = logging.getLogger(__name__)

from ._internal.configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from bentoml._internal.server.metrics.prometheus import PrometheusClient


class _MetricsMeta(ABC):
    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        if "registry" in kwargs:
            logger.warning(
                "'registry' shouldn't be passed as an argument as BentoML will handle multiprocess mode for prometheus metrics. Removing for now..."
            )
            kwargs.pop("registry")
        self._args = args
        self._kwargs = kwargs
        self._metric = None

    @abstractmethod
    def _setup_metric(
        self, metric_client: PrometheusClient = Provide[BentoMLContainer.metrics_client]
    ) -> None:
        raise NotImplementedError

    def __getattr__(self, item: t.Any):
        if self._metric is None:
            self._setup_metric()
        return getattr(self._metric, item)


class Histogram(_MetricsMeta):
    @inject
    def _setup_metric(
        self, metric_client: PrometheusClient = Provide[BentoMLContainer.metrics_client]
    ) -> None:
        self._metric = metric_client.Histogram(*self._args, **self._kwargs)


class Counter(_MetricsMeta):
    @inject
    def _setup_metric(
        self, metric_client: PrometheusClient = Provide[BentoMLContainer.metrics_client]
    ) -> None:
        self._metric = metric_client.Counter(*self._args, **self._kwargs)


class Summary(_MetricsMeta):
    @inject
    def _setup_metric(
        self, metric_client: PrometheusClient = Provide[BentoMLContainer.metrics_client]
    ) -> None:
        self._metric = metric_client.Summary(*self._args, **self._kwargs)


class Gauge(_MetricsMeta):
    @inject
    def _setup_metric(
        self, metric_client: PrometheusClient = Provide[BentoMLContainer.metrics_client]
    ) -> None:
        self._metric = metric_client.Gauge(*self._args, **self._kwargs)


class Info(_MetricsMeta):
    @inject
    def _setup_metric(
        self, metric_client: PrometheusClient = Provide[BentoMLContainer.metrics_client]
    ) -> None:
        self._metric = metric_client.Info(*self._args, **self._kwargs)


class Enum(_MetricsMeta):
    @inject
    def _setup_metric(
        self, metric_client: PrometheusClient = Provide[BentoMLContainer.metrics_client]
    ) -> None:
        self._metric = metric_client.Enum(*self._args, **self._kwargs)


class Metric(_MetricsMeta):
    @inject
    def _setup_metric(
        self, metric_client: PrometheusClient = Provide[BentoMLContainer.metrics_client]
    ) -> None:
        self._metric = metric_client.Metric(*self._args, **self._kwargs)
