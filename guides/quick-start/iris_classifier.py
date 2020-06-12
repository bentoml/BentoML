from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.artifact import SklearnModelArtifact


@env(auto_pip_dependencies=True)
@artifacts([SklearnModelArtifact('model')])
class IrisClassifier(BentoService):
    @api(input=DataframeInput())
    def predict(self, df):
        # Optional pre-processing, post-processing code goes here
        return self.artifacts.model.predict(df)
