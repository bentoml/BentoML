import numpy as np

import bentoml
from bentoml.adapters import JsonInput
from bentoml.frameworks.keras import KerasModelArtifact


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([KerasModelArtifact('model')])
class KerasClassifier(bentoml.BentoService):
    @bentoml.api(input=JsonInput(), batch=True)
    def predict(self, jsons):
        return self.artifacts.model(np.array(jsons))
