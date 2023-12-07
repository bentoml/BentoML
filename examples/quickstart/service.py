import bentoml_io as bentoml
import numpy as np


@bentoml.service(resources={"memory": "512MiB"})
class Preprocessing:
    @bentoml.api
    def preprocess(self, input_series: np.ndarray) -> np.ndarray:
        return input_series


@bentoml.service(resources={"memory": "512MiB"})
class IrisClassifier:
    iris_model = bentoml.models.get("iris_clf:latest")
    preprocessing = bentoml.depends(Preprocessing)

    def __init__(self):
        import joblib

        self.model = joblib.load(self.iris_model.path_of("model.pkl"))

    @bentoml.api
    def classify(self, input_series: np.ndarray) -> np.ndarray:
        input_series = self.preprocessing.preprocess(input_series)
        return self.model.predict(input_series)
