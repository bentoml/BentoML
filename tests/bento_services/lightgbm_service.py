import bentoml
from bentoml.lightgbm import LightGBMModelArtifact
from bentoml.adapters import DataframeInput


@bentoml.artifacts([LightGBMModelArtifact("model")])
@bentoml.env(infer_pip_packages=True)
class LgbModelService(bentoml.BentoService):
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        return self.artifacts.model.predict(df)
