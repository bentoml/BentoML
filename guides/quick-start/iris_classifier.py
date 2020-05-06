from bentoml import env, artifacts, api, BentoService
from bentoml.handlers import DataframeHandler
from bentoml.artifact import SklearnModelArtifact

@env(auto_pip_dependencies=True)
@artifacts([SklearnModelArtifact('model')])
class IrisClassifier(BentoService):

    @api(DataframeHandler)
    def predict(self, df):
        # Optional pre-processing, post-processing code goes here
        return self.artifacts.model.predict(df)
