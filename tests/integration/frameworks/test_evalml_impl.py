import evalml
import pandas as pd
import pytest

from bentoml.evalml import EvalMLModel
from tests._internal.helpers import assert_have_file_extension

test_df = pd.DataFrame([[42, "b"]])


@pytest.fixture(scope="session")
def binary_pipeline() -> "evalml.pipelines.BinaryClassificationPipeline":
    X = pd.DataFrame([[0, "a"], [0, "a"], [0, "a"], [42, "b"], [42, "b"], [42, "b"]])
    y = pd.Series([0, 0, 0, 1, 1, 1], name="target")
    pipeline = evalml.pipelines.BinaryClassificationPipeline(
        ["Imputer", "One Hot Encoder", "Random Forest Classifier"]
    )
    pipeline.fit(X, y)
    return pipeline


def test_evalml_save_load(tmpdir, binary_pipeline):
    EvalMLModel(binary_pipeline).save(tmpdir)
    assert_have_file_extension(tmpdir, ".pkl")

    evalml_loaded: "evalml.pipelines.PipelineBase" = EvalMLModel.load(tmpdir)
    assert (
        evalml_loaded.predict(test_df).to_numpy()
        == binary_pipeline.predict(test_df).to_numpy()
    )
