import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray


class IrisFeatures(BaseModel):
    sepal_len: float
    sepal_width: float
    petal_len: float
    petal_width: float


bento_model = bentoml.sklearn.get("iris_clf_with_feature_names:latest")
iris_clf_runner = bento_model.to_runner()

svc = bentoml.Service("iris_fastapi_demo", runners=[iris_clf_runner])


@svc.api(input=JSON(pydantic_model=IrisFeatures), output=NumpyNdarray())
async def predict_bentoml(input_data: IrisFeatures) -> np.ndarray:
    input_df = pd.DataFrame([input_data.dict()])
    return await iris_clf_runner.predict.async_run(input_df)


fastapi_app = FastAPI()
svc.mount_asgi_app(fastapi_app)


@fastapi_app.get("/metadata")
def metadata():
    return {"name": bento_model.tag.name, "version": bento_model.tag.version}


# For demo purpose, here's an identical inference endpoint implemented via FastAPI
@fastapi_app.post("/predict_fastapi")
def predict(features: IrisFeatures):
    input_df = pd.DataFrame([features.dict()])
    results = iris_clf_runner.predict.run(input_df)
    return {"prediction": results.tolist()[0]}


# For demo purpose, here's an identical inference endpoint implemented via FastAPI
@fastapi_app.post("/predict_fastapi_async")
async def predict_async(features: IrisFeatures):
    input_df = pd.DataFrame([features.dict()])
    results = await iris_clf_runner.predict.async_run(input_df)
    return {"prediction": results.tolist()[0]}
