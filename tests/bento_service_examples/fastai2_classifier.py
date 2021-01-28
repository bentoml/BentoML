import bentoml
import numpy as np
from bentoml.adapters import DataframeInput
from bentoml.frameworks.fastai import FastaiModelArtifact


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([FastaiModelArtifact('model')])
class FastaiClassifier(bentoml.BentoService):
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        input_data = df.to_numpy().astype(np.float32)
        _, _, output = self.artifacts.model.predict(input_data)

        return output.squeeze().item()
