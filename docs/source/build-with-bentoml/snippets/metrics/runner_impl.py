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
