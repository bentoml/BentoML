import pandas as pd
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler

import bentoml
from tests.bento_service_examples.pyspark_classifier import PysparkClassifier


train_pddf = pd.DataFrame([[0, -1.0], [1, 1.0]], columns=["label", "feature1"])
test_pddf = pd.DataFrame([-5.0, 5.0, -0.5, 0.5], columns=["feature1"])


def test_pyspark_model_pack(spark_session, tmpdir):
    # Put pandas training df into Spark df form with Vector features
    train_spdf = spark_session.createDataFrame(train_pddf)
    assembler = VectorAssembler(inputCols=['feature1'], outputCol='features')
    train_spdf = assembler.transform(train_spdf).select(['features', 'label'])

    # Train model (should result in x=neg => y=0, x=pos => y=1)
    lr = LogisticRegression()
    lr_model = lr.fit(train_spdf)

    # Test service with packed PySpark model
    svc = PysparkClassifier()
    svc.pack('model', lr_model)
    output_df = svc.predict(test_pddf)
    assert list(output_df.prediction) == [0.0, 1.0, 0.0, 1.0]

    # Test service that has been saved and loaded
    saved_dir = svc.save(str(tmpdir))
    loaded_svc = bentoml.load(saved_dir)
    output_df = loaded_svc.predict(test_pddf)
    assert list(output_df.prediction) == [0.0, 1.0, 0.0, 1.0]

