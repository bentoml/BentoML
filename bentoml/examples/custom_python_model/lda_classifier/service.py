import typing

import numpy as np

import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray

iris_clf_runner = bentoml.picklable_model.get("iris_clf_lda:latest").to_runner()

svc = bentoml.Service("iris_classifier_lda", runners=[iris_clf_runner])


@svc.api(input=NumpyNdarray(dtype="float", shape=(-1, 4)), output=JSON())
async def classify(input_series: np.ndarray) -> typing.List[float]:
    return await iris_clf_runner.predict.async_run(input_series)
