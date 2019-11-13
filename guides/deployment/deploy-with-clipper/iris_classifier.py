from bentoml import BentoService, api, env, artifacts
from bentoml.artifact import PickleArtifact
from bentoml.handlers import DataframeHandler, ClipperFloatsHandler

@artifacts([PickleArtifact('model')])
@env(pip_dependencies=["scikit-learn"])
class IrisClassifier(BentoService):

    @api(DataframeHandler)
    def predict(self, df):
        return self.artifacts.model.predict(df)
    
    @api(ClipperFloatsHandler)
    def predict_clipper(self, inputs):
        return self.artifacts.model.predict(inputs)