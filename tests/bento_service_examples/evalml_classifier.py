import pickle

import bentoml
from bentoml.adapters import DataframeInput
from bentoml.frameworks.evalml import EvalMLModelArtifact
from bentoml.utils import cloudpickle


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([EvalMLModelArtifact('model')])
class EvalMLClassifier(bentoml.BentoService):
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        return self.artifacts.model.predict(df).to_series().to_numpy()
