import pandas
import pytest

import evalml

import bentoml
from bentoml.yatai.client import YataiClient
from tests.bento_service_examples.evalml_classifier import EvalMLClassifier


@pytest.fixture(scope="session")
def evalml_pipeline():
    X = pandas.DataFrame(
        [[0, "a"], [0, "a"], [0, "a"], [42, "b"], [42, "b"], [42, "b"]]
    )
    y = pandas.Series([0, 0, 0, 1, 1, 1], name="target")
    pipeline = evalml.pipelines.BinaryClassificationPipeline(
        ["Imputer", "One Hot Encoder", "Random Forest Classifier"]
    )
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture()
def evalml_classifier_class():
    # When the ExampleBentoService got saved and loaded again in the test, the two class
    # attribute below got set to the loaded BentoService class. Resetting it here so it
    # does not effect other tests
    EvalMLClassifier._bento_service_bundle_path = None
    EvalMLClassifier._bento_service_bundle_version = None
    return EvalMLClassifier


test_df = pandas.DataFrame([[42, "b"]])


def test_evalml_artifact_pack(evalml_classifier_class, evalml_pipeline):
    svc = evalml_classifier_class()
    svc.pack("model", evalml_pipeline)
    assert svc.predict(test_df) == 1.0, "Run inference before save the artifact"

    saved_path = svc.save()
    loaded_svc = bentoml.load(saved_path)
    assert loaded_svc.predict(test_df) == 1.0, "Run inference from saved artifact"

    # clean up saved bundle
    yc = YataiClient()
    yc.repository.delete(f"{svc.name}:{svc.version}")
