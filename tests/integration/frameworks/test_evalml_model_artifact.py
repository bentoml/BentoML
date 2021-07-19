# ==============================================================================
#     Copyright (c) 2021 Atalaya Tech. Inc
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
# ==============================================================================

import os

import evalml
import pandas as pd
import pytest

from bentoml.evalml import EvalMLModel

from ..._internal.bento_services.evalml import mock_df


@pytest.fixture(scope="session")
def binary_pipeline() -> "evalml.pipelines.BinaryClassificationPipeline":
    X = pd.DataFrame([[0, 'a'], [0, 'a'], [0, 'a'], [42, 'b'], [42, 'b'], [42, 'b']])
    y = pd.Series([0, 0, 0, 1, 1, 1], name='target')
    pipeline = evalml.pipelines.BinaryClassificationPipeline(
        ['Imputer', 'One Hot Encoder', 'Random Forest Classifier']
    )
    pipeline.fit(X, y)
    return pipeline


def test_evalml_save_load(tmpdir, evalml_pipeline):
    EvalMLModel(evalml_pipeline, name="binary_classification_pipeline").save(tmpdir)
    assert os.path.exists(EvalMLModel.get_path(tmpdir, ".pkl"))

    evalml_loaded: "evalml.pipelines.PipelineBase" = EvalMLModel.load(tmpdir)
    assert (
        evalml_loaded.predict(mock_df).to_numpy()
        == evalml_pipeline.predict(mock_df).to_numpy()
    )
