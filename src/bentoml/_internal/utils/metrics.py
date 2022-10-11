from __future__ import annotations

INF = float("inf")

DEFAULT_BUCKET = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    INF,
)

MAX_BUCKET_COUNT = 100


def metric_name(*args: str | int) -> str:
    """
    Concatenates the given parts into a legal Prometheus metric name. For example,
    a valid tag name may includes invalid characters, so we need to escape them
    ref: https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels
    """
    return "_".join([str(arg).replace("-", ":").replace(".", "::") for arg in args])


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


def linear_buckets(start: float, step: float, end: float) -> tuple[float, ...]:
    """
    Creates buckets of a Prometheus histogram where the lowest bucket has an upper
    bound of start and the upper bound of each following bucket is the previous
    buckets upper bound plus step. The return tuple include the end as the second
    last value and positive infinity as the last value.
    """

    assert start > 0.0
    assert start < end
    assert step > 0.0

    bound = start
    buckets: list[float] = []
    while bound < end:
        buckets.append(bound)
        bound += step

    if len(buckets) > MAX_BUCKET_COUNT:
        buckets = buckets[:MAX_BUCKET_COUNT]

    return tuple(buckets) + (end, INF)
