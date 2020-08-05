import pytest

import pandas as pd
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler

import bentoml
from tests.bento_service_examples.pyspark_classifier import PysparkClassifier


train_pddf = pd.DataFrame([[0, -1.0], [1, 1.0]], columns=["label", "feature1"])
test_pddf = pd.DataFrame([-5.0, 5.0, -0.5, 0.5], columns=["feature1"])


@pytest.fixture()
def pyspark_model(spark_session):
    # Put pandas training df into Spark df form with Vector features
    train_spdf = spark_session.createDataFrame(train_pddf)
    assembler = VectorAssembler(inputCols=['feature1'], outputCol='features')
    train_spdf = assembler.transform(train_spdf).select(['features', 'label'])

    # Train model (should result in x=neg => y=0, x=pos => y=1)
    lr = LogisticRegression()
    lr_model = lr.fit(train_spdf)

    return lr_model


@pytest.fixture()
def pyspark_svc(pyspark_model):
    svc = PysparkClassifier()
    svc.pack('model', pyspark_model)

    return svc


@pytest.fixture()
def pyspark_svc_saved_dir(tmp_path_factory, pyspark_svc):
    """Save a TensorFlow2 BentoService and return the saved directory."""
    tmpdir = str(tmp_path_factory.mktemp("pyspark_svc"))
    pyspark_svc.save_to_dir(tmpdir)

    return tmpdir


@pytest.fixture()
def pyspark_svc_loaded(pyspark_svc_saved_dir):
    """Return a TensorFlow2 BentoService that has been saved and loaded."""
    return bentoml.load(pyspark_svc_saved_dir)


def test_pyspark_artifact(pyspark_svc):
    output_df = pyspark_svc.predict(test_pddf)
    assert list(output_df.prediction) == [0.0, 1.0, 0.0, 1.0]


def test_pyspark_artifact_loaded(pyspark_svc_loaded):
    output_df = pyspark_svc_loaded.predict(test_pddf)
    assert list(output_df.prediction) == [0.0, 1.0, 0.0, 1.0]

    # TODO: Test PySparkModelArtifact with Docker (like TensorFlow 2.0 tests)

