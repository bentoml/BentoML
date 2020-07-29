import pyspark
from pyspark.ml.classification import LogisticRegression


# TODO: Remove this demo test when PySparkSavedModelArtifact has been written
def test_spark_session_dataframe(spark_session):
    """Example code from spark-pytest repo. See:
    https://github.com/fdosani/travis-pytest-spark"""
    test_df = spark_session.createDataFrame([[1, 3], [2, 4]], "a: int, b: int")

    assert type(test_df) == pyspark.sql.dataframe.DataFrame
    assert test_df.count() == 2


# TODO: Remove this demo test when PySparkSavedModelArtifact has been written
def test_spark_session_sql(spark_session):
    """Example code from spark-pytest repo. See:
    https://github.com/fdosani/travis-pytest-spark"""
    test_df = spark_session.createDataFrame([[1, 3], [2, 4]], "a: int, b: int")
    test_df.registerTempTable('test')

    test_filtered_df = spark_session.sql('SELECT a, b from test where a > 1')
    assert test_filtered_df.count() == 1


# TODO: Remove this demo test when PySparkSavedModelArtifact has been written
def test_spark_mllib_example(spark_session):
    """Example code from Spark's MLLib main guide. See:
    https://spark.apache.org/docs/latest/ml-classification-regression.html"""
    # Load training data
    training = spark_session.read.format("libsvm").load("sample_libsvm_data.txt")

    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

    # Fit the model
    lrModel = lr.fit(training)

    # Print the coefficients and intercept for logistic regression
    print("Coefficients: " + str(lrModel.coefficients))
    print("Intercept: " + str(lrModel.intercept))

    # We can also use the multinomial family for binary classification
    mlr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8,
                             family="multinomial")

    # Fit the model
    mlrModel = mlr.fit(training)

    # Print the coefficients and intercepts for logistic regression with multinomial family
    print("Multinomial coefficients: " + str(mlrModel.coefficientMatrix))
    print("Multinomial intercepts: " + str(mlrModel.interceptVector))
