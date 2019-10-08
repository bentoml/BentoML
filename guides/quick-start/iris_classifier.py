from bentoml import BentoService, api, env, artifacts
from bentoml.artifact import SklearnModelArtifact
from bentoml.handlers import DataframeHandler

@artifacts([SklearnModelArtifact('model')])
@env(pip_dependencies=["scikit-learn"])
class IrisClassifier(BentoService):

    @api(DataframeHandler)
    def predict(self, df):
        return self.artifacts.model.predict(df)
