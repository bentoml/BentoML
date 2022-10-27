from __future__ import annotations

import bentoml

inference_duration = bentoml.metrics.Histogram(
    name="inference_duration",
    documentation="Duration of inference",
    labelnames=["nltk_version", "sentiment_cls"],
    buckets=exponential_buckets(0.001, 1.5, 10.0),
)

num_invocation = bentoml.metrics.Counter(
    name="num_invocation",
    documentation="Count total number of invocation for a given endpoint",
    labelnames=["endpoint"],
)
