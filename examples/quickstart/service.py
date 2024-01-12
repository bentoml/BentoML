import numpy as np
from typing_extensions import Annotated

import bentoml


@bentoml.service(resources={"cpu": "200m", "memory": "512Mi"})
class Preprocessing:
    @bentoml.api
    def preprocess(self, input_series: np.ndarray) -> np.ndarray:
        return input_series


@bentoml.service(name="iris_classifier", resources={"cpu": "200m", "memory": "512Mi"})
class IrisClassifier:
    iris_model = bentoml.models.get("iris_sklearn:latest")
    preprocessing = bentoml.depends(Preprocessing)

    def __init__(self):
        import joblib

        self.model = joblib.load(self.iris_model.path_of("model.pkl"))

    @bentoml.api
    def classify(
        self, input_series: Annotated[np.ndarray, bentoml.Shape((1, 4))]
    ) -> np.ndarray:
        input_series = self.preprocessing.preprocess(input_series)
        return self.model.predict(input_series)
