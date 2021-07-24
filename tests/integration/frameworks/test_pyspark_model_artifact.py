import pandas as pd
import pyspark.ml
import pytest
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

from bentoml.pyspark import SPARK_SESSION_NAMESPACE, PysparkMLlibModel

spark_session = SparkSession.builder.appName(SPARK_SESSION_NAMESPACE).getOrCreate()

train_pd_df = pd.DataFrame([[0, -1.0], [1, 1.0]], columns=["label", "feature1"])
test_pd_df = pd.DataFrame([-5.0, 5.0, -0.5, 0.5], columns=["feature1"])


def predict_df(model: pyspark.ml.Model, df: "pd.DataFrame"):
    spark_df = spark_session.createDataFrame(df)
    col_labels = [str(c) for c in list(df.columns)]
    assembler = VectorAssembler(inputCols=col_labels, outputCol="features")
    spark_df = assembler.transform(spark_df).select(["features"])
    output_df = model.transform(spark_df)
    return output_df.select("prediction").toPandas().prediction.values


@pytest.fixture(scope="module")
def pyspark_model():
    train_sp_df = spark_session.createDataFrame(train_pd_df)
    assembler = VectorAssembler(inputCols=["feature1"], outputCol="features")
    train_sp_df = assembler.transform(train_sp_df).select(["features", "label"])

    # train model (x=neg -> y=0, x=pos -> y=1)
    lr = LogisticRegression()
    model = lr.fit(train_sp_df)
    return model


def test_pyspark_mllib_save_load(
    tmpdir, pyspark_model
):  # pylint: disable=redefined-outer-name
    PysparkMLlibModel(pyspark_model).save(tmpdir)

    pyspark_loaded: pyspark.ml.Model = PysparkMLlibModel.load(tmpdir)

    assert (
        predict_df(pyspark_loaded, test_pd_df).tolist()
        == predict_df(pyspark_model, test_pd_df).tolist()
    )
