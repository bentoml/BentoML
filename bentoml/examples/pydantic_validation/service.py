import typing

import numpy as np
import pandas as pd
from pydantic import BaseModel

import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray

iris_clf_runner = bentoml.sklearn.get("iris_clf_with_feature_names:latest").to_runner()

svc = bentoml.Service("iris_classifier_pydantic", runners=[iris_clf_runner])


class IrisFeatures(BaseModel):
    sepal_len: float
    sepal_width: float
    petal_len: float
    petal_width: float

    # Optional field
    request_id: typing.Optional[int]

    # Use custom Pydantic config for additional validation options
    class Config:
        extra = "forbid"


input_spec = JSON(pydantic_model=IrisFeatures)


@svc.api(input=input_spec, output=NumpyNdarray())
async def classify(input_data: IrisFeatures) -> np.ndarray:
    if input_data.request_id is not None:
        print("Received request ID: ", input_data.request_id)

    input_df = pd.DataFrame([input_data.dict(exclude={"request_id"})])
    return await iris_clf_runner.predict.async_run(input_df)
