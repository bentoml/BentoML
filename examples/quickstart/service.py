import numpy as np

import bentoml
from bentoml.io import NumpyNdarray

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])


@svc.api(
    input=NumpyNdarray.from_sample(np.array([[4.9, 3.0, 1.4, 0.2]], dtype=np.double)),
    output=NumpyNdarray(),
)
def classify(input_series: np.ndarray) -> np.ndarray:
    return iris_clf_runner.predict.run(input_series)
