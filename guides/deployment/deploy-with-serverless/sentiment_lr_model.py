import pandas as pd
import bentoml
from bentoml.artifact import PickleArtifact
from bentoml.handlers import DataframeHandler

@bentoml.artifacts([PickleArtifact('sentiment_lr')])
@bentoml.env(pip_dependencies=['sklearn', 'numpy', 'pandas'])
class SentimentLRModel(bentoml.BentoService):
    
    @bentoml.api(DataframeHandler, typ='series')
    def predict(self, series):
        """
        predict expects pandas.Series as input
        """        
        return self.artifacts.sentiment_lr.predict(series)
