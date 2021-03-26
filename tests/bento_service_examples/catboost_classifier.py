import bentoml
from bentoml.adapters import DataframeInput
from bentoml.frameworks.catboost import CatBoostModelArtifact


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([CatBoostModelArtifact("model")])
class CatBoostModelClassifier(bentoml.BentoService):
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        return self.artifacts.model.predict(df)
