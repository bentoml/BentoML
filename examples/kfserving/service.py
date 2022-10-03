from typing import List

import numpy as np
import pydantic

import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])


class KFServingInputSchema(pydantic.BaseModel):
    instances: List[List[float]]


kfserving_input = JSON(pydantic_model=KFServingInputSchema)


@svc.api(
    input=kfserving_input,
    output=NumpyNdarray(),
    route="v1/models/iris_classifier",
)
async def classify(kf_input: KFServingInputSchema) -> np.ndarray:
    instances = np.array(kf_input.instances)
    return await iris_clf_runner.predict.async_run(instances)
