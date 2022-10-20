import numpy as np
import pandas as pd
import bentoml
from bentoml.io import PandasSeries, PandasDataFrame

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service("batch_processor", runners=[iris_clf_runner])

@svc.api(
    input=PandasDataFrame(),
    output=PandasDataFrame(),
)
async def classify1(input_series: pd.DataFrame) -> pd.DataFrame:
    return await PandasDataFrame(iris_clf_runner.predict.async_run(input_series))

@svc.api(
    input=PandasSeries(),
    output=PandasSeries(),
)
async def classify2(input_series: pd.Series) -> pd.Series:
    return await PandasSeries(iris_clf_runner.predict.async_run(input_series))
