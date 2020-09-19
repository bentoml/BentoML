import pandas as pd

from bentoml import BentoService, api, artifacts, env
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact


@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('model')])
class IrisClassifier(BentoService):
    @api(input=DataframeInput(), batch=True)
    def predict(self, df: pd.DataFrame):
        # Optional pre-processing, post-processing code goes here
        return self.artifacts.model.predict(df)
