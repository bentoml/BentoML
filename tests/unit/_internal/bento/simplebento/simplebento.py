import bentoml
import pyspark
import pandas as pd
from bentoml.io import PandasSeries

svc = bentoml.Service("test.simplebento")

@svc.api(input=PandasSeries(), output=PandasSeries())
def increment(input: pd.Series[int]) -> pd.Series[int]:
    for i in input:
        i += 1
    return input

@svc.api(input=PandasSeries(), output=PandasSeries())
def uppercase(input: pd.Series[str]) -> pd.Series[str]:
    for i in input:
        i.upper()
    return input