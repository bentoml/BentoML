import pandas as pd
import bentoml
from bentoml.artifact import PickleArtifact
from bentoml.handlers import DataframeHandler

@bentoml.artifacts([PickleArtifact('sentiment_lr')])
@bentoml.env(conda_dependencies=["scikit-learn", "pandas"])
class SentimentLRModel(bentoml.BentoService):

    @bentoml.api(DataframeHandler)
    def predict(self, df):
        """
        predict expects dataframe as input
        """        
        return self.artifacts.sentiment_lr.predict(df)
