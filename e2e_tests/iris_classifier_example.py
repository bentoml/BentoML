from bentoml import BentoService, api, env, artifacts
from bentoml.artifact import SklearnModelArtifact
from bentoml.adapters import DataframeInput


@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('clf')])
class IrisClassifier(BentoService):
    @api(input=DataframeInput())
    def predict(self, df):
        return self.artifacts.clf.predict(df)
