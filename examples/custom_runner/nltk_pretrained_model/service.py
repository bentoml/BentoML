import bentoml
import nltk
from bentoml.io import Text, JSON
from nltk.sentiment import SentimentIntensityAnalyzer
from statistics import mean


class NLTKSentimentAnalysisRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self): 
        self.sia = SentimentIntensityAnalyzer()

    @bentoml.Runnable.method(batchable=False)
    def is_positive(self, input_text):
        scores = [
            self.sia.polarity_scores(sentence)["compound"]
            for sentence in nltk.sent_tokenize(input_text)
        ]
        return mean(scores) > 0

nltk_runner = bentoml.Runner(NLTKSentimentAnalysisRunnable, name='nltk_sentiment')

svc = bentoml.Service('sentiment_analyzer', runners=[nltk_runner])

@svc.api(input=Text(), output=JSON())
def analysis(input_text):
    is_positive = nltk_runner.is_positive.run(input_text)
    return { "is_positive": is_positive }
