import typing

import bentoml
from bentoml.io import NumpyNdarray

if typing.TYPE_CHECKING:
    import numpy as np

agaricus_runner = bentoml.xgboost.get("agaricus:latest").to_runner()

svc = bentoml.Service("agaricus", runners=[agaricus_runner])


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def classify(input_data: "np.ndarray") -> "np.ndarray":
    return await agaricus_runner.async_run(input_data)
