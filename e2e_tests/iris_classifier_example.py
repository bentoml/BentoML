from bentoml import BentoService, api, env, artifacts
from bentoml.artifact import SklearnModelArtifact
from bentoml.handlers import DataframeHandler


@env(auto_pip_dependencies=True)
@artifacts([SklearnModelArtifact('clf')])
class IrisClassifier(BentoService):
    @api(DataframeHandler)
    def predict(self, df):
        return self.artifacts.clf.predict(df)
