from bentoml import BentoService, api, env, artifacts
from bentoml.artifact import PickleArtifact
from bentoml.handlers import DataframeHandler


@env(auto_pip_dependencies=True)
@artifacts([PickleArtifact('clf')])
class IrisClassifier(BentoService):
    @api(DataframeHandler)
    def predict(self, df):
        return self.artifacts.clf.predict(df)
