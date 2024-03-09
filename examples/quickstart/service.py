import numpy as np
from typing_extensions import Annotated

import bentoml
from bentoml.validators import Shape


@bentoml.service(name="preprocessing", resources={"cpu": "200m", "memory": "512Mi"})
class Preprocessing:
    @bentoml.api
    def preprocess(self, input_series: np.ndarray) -> np.ndarray:
        return input_series


@bentoml.service(resources={"cpu": "200m", "memory": "512Mi"})
class IrisClassifier:
    iris_model = bentoml.models.get("iris_sklearn:latest")
    preprocessing = bentoml.depends(Preprocessing)

    def __init__(self):
        import joblib

        self.model = joblib.load(self.iris_model.path_of("model.pkl"))

    @bentoml.api
    def classify(
        self, input_series: Annotated[np.ndarray, Shape((1, 4))]
    ) -> np.ndarray:
        input_series = self.preprocessing.preprocess(input_series)
        return self.model.predict(input_series)


if __name__ == "__main__":
    server = IrisClassifier.serve_http(threaded=True)
    try:
        with bentoml.SyncHTTPClient(
            "http://localhost:3000", server_ready_timeout=10
        ) as client:
            response = client.classify([[5.1, 3.5, 1.4, 0.2]])
            print(response)
    finally:
        server.stop()
