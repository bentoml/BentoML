import numpy as np

import bentoml
import xgboost as xgb  # pylint: disable=import-error
from bentoml.adapters import DataframeInput
from bentoml.xgboost import XgboostModelArtifact


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([XgboostModelArtifact("model")])
class XgboostModelClassifier(bentoml.BentoService):
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        dmatrix = xgb.DMatrix(df)
        result = self.artifacts.model.predict(dmatrix)
        preds = np.asarray([np.argmax(line) for line in result])
        return preds
