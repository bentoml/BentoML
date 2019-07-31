import xgboost as xgb

import bentoml
from bentoml.artifact import XgboostModelArtifact
from bentoml.handlers import DataframeHandler

@bentoml.artifacts([XgboostModelArtifact('model')])
@bentoml.env(pip_dependencies=['xgboost'])
class TitanicModel(bentoml.BentoService):
    
    @bentoml.api(DataframeHandler)
    def predict(self, df):
        data = xgb.DMatrix(data=df[['Pclass', 'Age', 'Fare', 'SibSp', 'Parch']])
        return self.artifacts.model.predict(data)
