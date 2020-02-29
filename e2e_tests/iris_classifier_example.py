from bentoml import BentoService, api, env, artifacts
from bentoml.artifact import PickleArtifact
from bentoml.handlers import DataframeHandler


@artifacts([PickleArtifact('clf')])
@env(pip_dependencies=['scikit-learn'])
class IrisClassifier(BentoService):
    @api(DataframeHandler)
    def predict(self, df):
        return self.artifacts.clf.predict(df)
