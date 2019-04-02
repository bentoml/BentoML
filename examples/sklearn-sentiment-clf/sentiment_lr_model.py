import bentoml
import pandas as pd

class SentimentLRModel(bentoml.BentoModel):
    """
    My SentimentLRModel packaging with BentoML
    """
    
    def config(self, artifacts, env):
        artifacts.add(bentoml.PickleArtifact('sentiment_lr'))

        env.add_conda_dependencies(["scikit-learn", "pandas"])
        
        
    def predict(self, df):
        """
        predict expects dataframe as input
        """
        return self.artifacts.sentiment_lr.predict(df)
