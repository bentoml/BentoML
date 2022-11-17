import pandas as pd
import pytest
from pyspark.sql.session import SparkSession

import bentoml._internal.spark
from bentoml._internal.spark import _distribute_bento
from bentoml._internal.bento.bento import Bento
from bentoml._internal.bento.bento import BentoStore
from bentoml._internal.configuration.containers import BentoMLContainer


def test_simplebento(
    spark: SparkSession, simplebento: Bento, bento_store: BentoStore
) -> None:

    # get properties of the simple bento before it gets distributed
    bento_tag = simplebento.info.tag
    bento_service = simplebento.info.service
    bento_apis = simplebento.info.apis

    BentoMLContainer.bento_store.set(bento_store)

    # distribute bento file to the worker nodes
    _distribute_bento(spark, simplebento)

    data1: pd.Series[int] = pd.Series([1, 2, 3])
    df1 = spark.createDataFrame(pd.DataFrame(data1, columns=["data"]))
    df1 = bentoml._internal.spark.run_in_spark(spark, df1, simplebento, "increment")
    pd.testing.assert_frame_equal(
        df1.select("*").toPandas(), pd.DataFrame([2, 3, 4], columns=["data"])
    )

    # compare properties between the distributed and returned bentos
    assert bento_tag == simplebento.info.tag
    assert bento_service == simplebento.info.service
    assert bento_apis == simplebento.info.apis

    # test that passing in a multi-column df would cause an exception
    data2 = [["a", 1], ["b", 2], ["c", 3]]
    df2 = spark.createDataFrame(pd.DataFrame(data2, columns=["char", "int"]))
    df2 = bentoml._internal.spark.run_in_spark(spark, df2, simplebento, "increment")
    with pytest.raises(Exception):
        assert df2.select("*") == pd.DataFrame([["a", 2], ["b", 3], ["c", 4]])
