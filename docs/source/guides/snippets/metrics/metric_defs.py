from __future__ import annotations

import bentoml

inference_duration = bentoml.metrics.Histogram(
    name="inference_duration",
    documentation="Duration of inference",
    labelnames=["nltk_version", "sentiment_cls"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float("inf")),
)

num_invocation = bentoml.metrics.Counter(
    name="num_invocation",
    documentation="Count total number of invocation for a given endpoint",
    labelnames=["endpoint"],
)
