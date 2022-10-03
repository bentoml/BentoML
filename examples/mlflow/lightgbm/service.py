import numpy as np

import bentoml
from bentoml.io import NumpyNdarray

lgb_iris_runner = bentoml.mlflow.get("lgb_iris:latest").to_runner()

svc = bentoml.Service("lgb_iris_service", runners=[lgb_iris_runner])

input_spec = NumpyNdarray(
    dtype="float64",
    enforce_dtype=True,
    shape=(-1, 4),
    enforce_shape=True,
)


@svc.api(input=input_spec, output=NumpyNdarray())
async def classify(input_series: np.ndarray) -> np.ndarray:
    return await lgb_iris_runner.predict.async_run(input_series)
