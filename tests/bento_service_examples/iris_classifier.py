import bentoml
from bentoml.adapters import DataframeInput
from bentoml.artifact import SklearnModelArtifact


@bentoml.env(auto_pip_dependencies=True)
@bentoml.artifacts([SklearnModelArtifact('model')])
class IrisClassifier(bentoml.BentoService):
    @bentoml.api(input=DataframeInput())
    def predict(self, df):
        return self.artifacts.model.predict(df)
