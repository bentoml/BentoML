import bentoml
from bentoml.adapters import DataframeInput
from bentoml.frameworks.xgboost import XgboostModelArtifact
import xgboost as xgb  # pylint: disable=import-error
import numpy as np


@bentoml.env(auto_pip_dependencies=True)
@bentoml.artifacts([XgboostModelArtifact('model')])
class XgboostModelClassifier(bentoml.BentoService):
    @bentoml.api(input=DataframeInput())
    def predict(self, df):
        dmatrix = xgb.DMatrix(df)
        result = self.artifacts.model.predict(dmatrix)
        preds = np.asarray([np.argmax(line) for line in result])
        return preds
