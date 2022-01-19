import numpy as np

import bentoml
from bentoml.adapters import JsonInput
from bentoml.frameworks.keras import KerasModelArtifact


@bentoml.env(pip_packages=["keras==2.6.0", "tensorflow==2.6.2"])
@bentoml.artifacts(
    [
        KerasModelArtifact("model"),
        # TODO: #1698 set store_as_json_and_weights to True after the issue is fixed
        KerasModelArtifact("model2", store_as_json_and_weights=False),
    ]
)
class KerasClassifier(bentoml.BentoService):
    @bentoml.api(input=JsonInput(), batch=True)
    def predict(self, jsons):
        return self.artifacts.model.predict(np.array(jsons))

    @bentoml.api(input=JsonInput(), batch=True)
    def predict2(self, jsons):
        return self.artifacts.model2.predict(np.array(jsons))
