from bentoml import BentoService, api, env, artifacts
from bentoml.artifact import PickleArtifact
from bentoml.handlers import DataframeHandler

# You can also import your own python module here and BentoML will automatically
# figure out the dependency chain and package all those python modules

@artifacts([PickleArtifact('model')])
@env(pip_dependencies=["scikit-learn"])
class IrisClassifier(BentoService):

    @api(DataframeHandler)
    def predict(self, df):
        # arbitrary preprocessing or feature fetching code can be placed here 
        return self.artifacts.model.predict(df)
