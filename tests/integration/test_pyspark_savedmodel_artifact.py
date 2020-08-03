import pyspark
from pyspark.ml.classification import LogisticRegression

import bentoml
from tests.bento_service_examples.pyspark_classifier import PysparkClassifier


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


def test_pyspark_model_pack(spark_session, tmpdir):
    from pyspark.ml.feature import VectorAssembler
    vector_assembler = VectorAssembler(inputCols=['val'], outputCol='features')

    # Create toy training df
    train_df = spark_session.createDataFrame([[0, -1], [1, 1]],
                                             "label: int, val: int")
    train_df = vector_assembler.transform(train_df)
    train_df = train_df.select(['features', 'label'])

    # Create toy testing df
    test_df = spark_session.createDataFrame([[0, -5], [1, 5]],
                                            "label: int, val: int")
    test_df = vector_assembler.transform(test_df)
    test_df = test_df.select(['features', 'label'])

    # Train model
    lr = LogisticRegression()
    lr_model = lr.fit(train_df)

    # Check predictions
    lr_predictions = lr_model.transform(test_df)
    pred_list = lr_predictions.select("prediction").toPandas().values.tolist()
    assert pred_list == [[0.0], [1.0]]

    svc = PysparkClassifier()
    svc.pack('model', lr_model)
    saved_dir = svc.save(str(tmpdir))
    loaded_svc = bentoml.load(saved_dir)

