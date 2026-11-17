from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

import bentoml

PORT = 35681


def post_prediction(
    base_url: str,
    value: int,
    latency: int,
) -> list[int]:
    with httpx.Client(base_url=base_url, timeout=30) as client:
        response = client.post(
            "/predict",
            json={"values": [value]},
            headers={"X-Latency": str(latency)},
        )
        response.raise_for_status()
        return response.json()


def test_dynamic_latency_survives_build_and_shares_batches(examples: Path) -> None:
    fixture = examples / "dynamic_batching"
    bento = bentoml.build(
        "service.py:DynamicBatching",
        build_ctx=str(fixture),
    )

    with bentoml.serve(bento, port=PORT) as server:
        with bentoml.SyncHTTPClient(
            server.url,
            server_ready_timeout=100,
        ) as client:
            with ThreadPoolExecutor(max_workers=16) as executor:
                warmup = [
                    executor.submit(post_prediction, server.url, value, 500)
                    for value in range(16)
                ]
                assert [future.result() for future in warmup] == [
                    [value] for value in range(16)
                ]

            client.reset()

            values = [100, 101, 102, 103]
            latencies = [500, 500, 50, 500]
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(post_prediction, server.url, value, latency)
                    for value, latency in zip(values, latencies)
                ]
                assert [future.result() for future in futures] == [
                    [value] for value in values
                ]

            stats = client.stats()
            assert sorted(stats["latencies"]) == sorted(latencies)
            assert any(
                102 in batch and len(set(values).intersection(batch)) >= 2
                for batch in stats["batches"]
            )
