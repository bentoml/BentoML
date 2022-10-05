from __future__ import annotations

import typing as t
import logging

logger = logging.getLogger(__name__)

from ._internal.configuration.containers import BentoMLContainer


def __dir__() -> list[str]:
    metrics_client = BentoMLContainer.metrics_client.get()
    return dir(metrics_client.prometheus_client)


def __getattr__(item: t.Any):
    metrics_client = BentoMLContainer.metrics_client.get()
    if item in dir(metrics_client):
        return getattr(metrics_client, item)
    return getattr(metrics_client.prometheus_client, item)
