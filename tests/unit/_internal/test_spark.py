import typing as t
import pytest
from pyspark.sql.session import SparkSession
import pandas as pd

import bentoml
from bentoml._internal.tag import Tag
from bentoml.bentos import get as get_bento
from bentoml.io import PandasSeries, PandasDataFrame
from bentoml._internal.spark import _distribute_bento, _load_bento, _get_process # ignore errors?

# create one spark session for the whole test session
@pytest.fixture(scope="session")
def spark_session():
    return SparkSession.builder.master("local[1]").appName("testApp").getOrCreate()

# initialize a bentoml service and api for testing
iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()
svc = bentoml.Service("test_spark", runners=[iris_clf_runner])

# get bento model tag
assert svc.tag is not None # so Pylance doesn't complain
bento_tag: Tag = svc.tag

@svc.api(
    input=PandasDataFrame(),
    output=PandasSeries(dtype="float"),
)
def spark_test_api(input_series: pd.DataFrame) -> pd.Series[t.Any]:
    print(input_series)
    return pd.Series(iris_clf_runner.predict.run(input_series))
api_name = "spark_test_api"

# test each function in spark.py:

def test_distribute_bento(spark_session: SparkSession) -> None:    
    assert _distribute_bento(spark_session, get_bento(bento_tag)) \
        == f"{bento_tag.name}-{bento_tag.version}.bento"

def test_load_bento() -> None:
    assert _load_bento(bento_tag) == svc

def test_get_process() -> None:
    for runner in svc.runners:
        runner.init_local(quiet=True)
    assert (
            api_name in svc.apis
        ), "An error occurred transferring the Bento to the Spark worker"
    inference_api = svc.apis[api_name]
    assert inference_api.func is not None, "Inference API function not defined"
    assert callable(_get_process(bento_tag, api_name)) # check that it returns a function

def test_get_udf(spark_session: SparkSession) -> None:
    None


