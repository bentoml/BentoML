import numpy as np

import bentoml_io as bentoml


@bentoml.service(resources={"num_cpus": 1})
class IrisClassifier:
    iris_model = bentoml.models.get("iris_clf:latest")

    def __init__(self):
        import joblib

        self.model = joblib.load(self.iris_model.path_of("model.pkl"))

    @bentoml.api
    def classify(self, input_series: np.ndarray) -> np.ndarray:
        return self.model.predict(input_series)
