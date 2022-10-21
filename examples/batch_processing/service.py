from __future__  import annotations

import pandas as pd
import bentoml
from bentoml.io import PandasSeries, PandasDataFrame

import typing as t

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service("batch_processor", runners=[iris_clf_runner])

@svc.api(
    input=PandasDataFrame(),
    output=PandasSeries(dtype="float"),
)
def classify1(input_series: pd.DataFrame) -> pd.Series[t.Any]:
    print(input_series)
    return pd.Series(iris_clf_runner.predict.run(input_series))
