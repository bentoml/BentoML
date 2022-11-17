from __future__ import annotations

import pandas as pd
import pyspark

import bentoml
from bentoml.io import PandasSeries

svc = bentoml.Service("test.simplebento")


@svc.api(input=PandasSeries(), output=PandasSeries(dtype="int"))
def increment(series: pd.Series[int]) -> pd.Series[int]:
    series += 1
    return series
