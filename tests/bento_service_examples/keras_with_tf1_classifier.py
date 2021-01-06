import numpy as np

import bentoml
from bentoml.adapters import JsonInput
from bentoml.frameworks.keras import KerasModelArtifact


@bentoml.env(pip_packages=["keras==2.3.1", "tensorflow==1.14", "h5py==2.10.0"])
@bentoml.artifacts(
    [
        KerasModelArtifact('model'),
        KerasModelArtifact('model2', store_as_json_and_weights=True),
    ]
)
class KerasClassifier(bentoml.BentoService):
    @bentoml.api(input=JsonInput(), batch=True)
    def predict(self, jsons):
        return self.artifacts.model.predict(np.array(jsons))

    @bentoml.api(input=JsonInput(), batch=True)
    def predict2(self, jsons):
        return self.artifacts.model2.predict(np.array(jsons))
