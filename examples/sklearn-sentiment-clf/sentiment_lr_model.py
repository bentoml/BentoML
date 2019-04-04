import pandas as pd
import bentoml
from bentoml.artifacts import PickleArtifact

class SentimentLRModel(bentoml.BentoModel):
    
    def config(self, artifacts, env):
        artifacts.add(PickleArtifact('sentiment_lr'))

        env.add_conda_dependencies(["scikit-learn", "pandas"])

    @bentoml.api(bentoml.handlers.DataframeHandler)
    def predict(self, df):
        """
        predict expects dataframe as input
        """        
        return self.artifacts.sentiment_lr.predict(df)
