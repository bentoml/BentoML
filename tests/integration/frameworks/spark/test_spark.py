import pytest

from bentoml._internal.bento.bento import Bento
import bentoml._internal.spark
from bentoml._internal.spark import _distribute_bento, _load_bento

from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col

import pandas as pd

def test_simplebento1(spark: SparkSession, simplebento1: Bento) -> None:

    # get properties of the simple bento before it gets distributed 
    bento_tag = simplebento1.info.tag
    bento_service = simplebento1.info.service
    bento_apis = simplebento1.info.apis

    # distribute bento file to the worker nodes
    _distribute_bento(spark, simplebento1)

    # create df
    data: pd.Series[int] = pd.Series([1, 2, 3])
    df = spark.createDataFrame(pd.DataFrame(data, columns=["data"]))

    # create a udf from the service api function
    udf = bentoml._internal.spark.get_udf(spark, "test.simplebento:latest", "increment")

    # apply the udf on the df and retrieve it to driver program
    df.select(udf(col("data"))).show()
   
   # load bento from the worker nodes
    _load_bento(simplebento1.tag)

    # compare properties between the distributed and returned bentos
    assert bento_tag == simplebento1.info.tag
    assert bento_service == simplebento1.info.service
    assert bento_apis == simplebento1.info.apis

def test_simplebento2(spark: SparkSession, simplebento2: Bento) -> None:

    # get properties of the simple bento before it gets distributed 
    bento_tag = simplebento2.info.tag
    bento_service = simplebento2.info.service
    bento_apis = simplebento2.info.apis

    # distribute bento file to the worker nodes
    _distribute_bento(spark, simplebento2)

    # create df
    data: pd.Series[int] = pd.Series(["a", "b", "c"])
    df = spark.createDataFrame(pd.DataFrame(data, columns=["data"]))

    # create a udf from the service api function
    udf = bentoml._internal.spark.get_udf(spark, "test.simplebento:latest", "uppercase")

    # apply the udf on the df and retrieve it to driver program
    df.select(udf(col("data"))).show()
   
   # load bento from the worker nodes
    _load_bento(simplebento2.tag)

    # compare properties between the distributed and returned bentos
    assert bento_tag == simplebento2.info.tag
    assert bento_service == simplebento2.info.service
    assert bento_apis == simplebento2.info.apis



