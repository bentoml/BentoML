import numpy as np

import bentoml


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class IrisClassifier:
    bento_model = bentoml.models.get("iris:latest")

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

    @bentoml.api
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        rv = self.model.predict(input_data)
        return np.asarray(rv)
