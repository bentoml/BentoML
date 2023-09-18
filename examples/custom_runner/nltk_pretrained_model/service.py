from __future__ import annotations

import time
import typing as t
from typing import TYPE_CHECKING
from statistics import mean

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

import bentoml
from bentoml.io import JSON
from bentoml.io import Text

if TYPE_CHECKING:
    from bentoml._internal.runner.runner import RunnerMethod

    class RunnerImpl(bentoml.Runner):
        is_positive: RunnerMethod


inference_duration = bentoml.metrics.Histogram(
    name="inference_duration",
    documentation="Duration of inference",
    labelnames=["nltk_version", "sentiment_cls"],
    buckets=(
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
        float("inf"),
    ),
)

polarity_counter = bentoml.metrics.Counter(
    name="polarity_total",
    documentation="Count total number of analysis by polarity scores",
    labelnames=["polarity"],
)


class NLTKSentimentAnalysisRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    @bentoml.Runnable.method(batchable=False)
    def is_positive(self, input_text: str) -> bool:
        start = time.perf_counter()
        scores = [
            self.sia.polarity_scores(sentence)["compound"]
            for sentence in nltk.sent_tokenize(input_text)
        ]
        inference_duration.labels(
            nltk_version=nltk.__version__, sentiment_cls=self.sia.__class__.__name__
        ).observe(time.perf_counter() - start)
        return mean(scores) > 0


nltk_runner = t.cast(
    "RunnerImpl", bentoml.Runner(NLTKSentimentAnalysisRunnable, name="nltk_sentiment")
)

svc = bentoml.Service("sentiment_analyzer", runners=[nltk_runner])


@svc.api(input=Text(), output=JSON())
async def analysis(input_text: str) -> dict[str, bool]:
    is_positive = await nltk_runner.is_positive.async_run(input_text)
    polarity_counter.labels(polarity=is_positive).inc()
    return {"is_positive": is_positive}
