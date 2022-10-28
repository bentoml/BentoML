from __future__ import annotations

INF = float("inf")

MAX_BUCKET_COUNT = 100


def exponential_buckets(start: float, factor: float, end: float) -> tuple[float, ...]:
    """
    Creates buckets of a Prometheus histogram where the lowest bucket has an upper
    bound of start and the upper bound of each following bucket is factor times the
    previous buckets upper bound. The return tuple include the end as the second
    last value and positive infinity as the last value.
    """

    assert start > 0.0
    assert start < end
    assert factor > 1.0

    bound = start
    buckets: list[float] = []
    while bound < end:
        buckets.append(bound)
        bound *= factor

    if len(buckets) > MAX_BUCKET_COUNT:
        buckets = buckets[:MAX_BUCKET_COUNT]

    return tuple(buckets) + (end, INF)
