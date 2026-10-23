from __future__ import annotations

import threading
from typing import Any

from starlette.requests import Request

import bentoml

RESOLVED_LATENCIES: list[int] = []
LOCK = threading.Lock()


def resolve_latency(request: Request) -> int:
    value = int(request.headers.get("X-Latency", "100"))
    with LOCK:
        RESOLVED_LATENCIES.append(value)
    return value


@bentoml.service(metrics={"enabled": False}, workers=1)
class DynamicBatching:
    def __init__(self) -> None:
        self.batches: list[list[int]] = []

    @bentoml.api(
        batchable=True,
        max_batch_size=8,
        max_latency_ms=resolve_latency,
    )
    def predict(self, values: list[int]) -> list[int]:
        with LOCK:
            self.batches.append(list(values))
        return values

    @bentoml.api
    def reset(self) -> bool:
        with LOCK:
            self.batches.clear()
            RESOLVED_LATENCIES.clear()
        return True

    @bentoml.api
    def stats(self) -> dict[str, Any]:
        with LOCK:
            return {
                "batches": list(self.batches),
                "latencies": list(RESOLVED_LATENCIES),
            }
