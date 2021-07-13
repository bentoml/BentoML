from bentoml import BentoService, api, artifacts, env
from bentoml.adapters import DataframeInput
from bentoml.sklearn import SklearnModelArtifact


@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('clf')])
class IrisClassifier(BentoService):
    @api(input=DataframeInput(), batch=True)
    def predict(self, df):
        return self.artifacts.clf.predict(df)
