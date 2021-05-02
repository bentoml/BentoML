import pickle

import bentoml
from bentoml.adapters import DataframeInput
from bentoml.frameworks.evalml import EvalMLModelArtifact
from bentoml.utils import cloudpickle


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([EvalMLModelArtifact('model', pickle_module=pickle)])
class EvalMLClassifier(bentoml.BentoService):
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        return self.artifacts.model.predict(df).to_series().to_numpy()


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([EvalMLModelArtifact('model', pickle_module=cloudpickle)])
class EvalMLClassifierCloudpickle(bentoml.BentoService):
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        return self.artifacts.model.predict(df).to_series().to_numpy()
