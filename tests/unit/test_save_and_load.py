#  Copyright (c) 2021 Atalaya Tech, Inc.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==========================================================================
#

import json
import logging
import os
import uuid

import mock
import pytest
from mock import patch

import bentoml
from bentoml.adapters import DataframeInput
from bentoml.exceptions import BentoMLException
from bentoml.saved_bundle import load_bento_service_metadata
from bentoml.utils.open_api import get_open_api_spec_json
from tests.unit.conftest import delete_saved_bento_service


class TestModel(object):
    def predict(self, input_data):
        return int(input_data) * 2


def test_save_and_load_model(tmpdir, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=10)(
        example_bento_service_class
    )

    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack("model", test_model)

    assert svc.predict(1000) == 2000
    version = "test_" + uuid.uuid4().hex

    svc.save_to_dir(str(tmpdir), version=version)
    model_service = bentoml.load(str(tmpdir))

    expected_version = "2.10.{}".format(version)
    assert model_service.version == expected_version

    api = model_service.get_inference_api("predict")
    assert api.name == "predict"
    assert api.batch
    assert api.mb_max_latency == 1000
    assert api.mb_max_batch_size == 2000
    assert isinstance(api.input_adapter, DataframeInput)
    assert api.user_func(1, tasks=[]) == 2

    # Check api methods are available
    assert model_service.predict(1) == 2


def test_warning_when_save_without_declared_artifact(
    tmpdir, example_bento_service_class
):
    svc = example_bento_service_class()

    with mock.patch("bentoml.saved_bundle.bundler.logger") as log_mock:
        svc.save_to_dir(str(tmpdir))
        log_mock.warning.assert_called_once_with(
            "Missing declared artifact '%s' for BentoService '%s'",
            "model",
            "ExampleBentoService",
        )


def test_pack_on_bento_service_instance(tmpdir, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=10)(
        example_bento_service_class
    )
    test_model = TestModel()
    svc = example_bento_service_class()

    svc.pack("model", test_model)
    assert svc.predict(1000) == 2000

    version = "test_" + uuid.uuid4().hex
    svc.set_version(version)

    svc.save_to_dir(str(tmpdir))
    model_service = bentoml.load(str(tmpdir))

    expected_version = "2.10.{}".format(version)
    assert model_service.version == expected_version

    api = model_service.get_inference_api("predict")
    assert api.name == "predict"
    assert isinstance(api.input_adapter, DataframeInput)
    assert api.user_func(1) == 2
    # Check api methods are available
    assert model_service.predict(1) == 2


def test_pack_metadata_invalid(example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=10)(
        example_bento_service_class
    )
    test_model = TestModel()
    svc = example_bento_service_class()

    # assert empty metadata before packing
    assert svc.artifacts.get("model").metadata == {}

    # try packing invalid
    model_metadata = "non-dictionary metadata"

    with pytest.raises(TypeError):
        svc.pack("model", test_model, metadata=model_metadata)


def test_pack_metadata(tmpdir, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=10)(
        example_bento_service_class
    )
    test_model = TestModel()
    svc = example_bento_service_class()

    model_metadata = {
        "k1": "v1",
        "job_id": "ABC",
        "score": 0.84,
        "datasets": ["A", "B"],
    }
    svc.pack("model", test_model, metadata=model_metadata)

    # check saved metadata is correct
    assert svc.artifacts.get("model").metadata == model_metadata

    svc.save_to_dir(str(tmpdir))
    model_service = bentoml.load(str(tmpdir))

    # check loaded metadata is correct
    assert model_service.artifacts.get("model").metadata == model_metadata


def test_open_api_spec_json(tmpdir, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=10)(
        example_bento_service_class
    )
    test_model = TestModel()
    svc = example_bento_service_class()

    svc.pack("model", test_model)
    before_json_d = get_open_api_spec_json(svc)

    svc.save_to_dir(str(tmpdir))
    with open(os.path.join(str(tmpdir), "docs.json")) as f:
        after_json_d = json.load(f)

    # check loaded json dictionary is the same as before saving
    assert before_json_d == after_json_d


class TestBentoWithOutArtifact(bentoml.BentoService):
    __test__ = False

    @bentoml.api(input=DataframeInput(), batch=True)
    def test(self, df):
        return df


def test_bento_without_artifact(tmpdir):
    TestBentoWithOutArtifact().save_to_dir(str(tmpdir))
    model_service = bentoml.load(str(tmpdir))
    assert model_service.test(1) == 1
    assert len(model_service.inference_apis) == 1


def test_save_duplicated_bento_exception_raised(example_bento_service_class):
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack("model", test_model)

    saved_path = svc.save()
    svc_metadata = load_bento_service_metadata(saved_path)
    assert svc.version == svc_metadata.version

    with pytest.raises(BentoMLException):
        with patch.object(bentoml.BentoService, "save_to_dir") as save_to_dir_method:
            # attempt to save again
            svc.save()
            save_to_dir_method.assert_not_called()

    # reset svc version
    svc.set_version()
    saved_path = svc.save()
    svc_metadata_new = load_bento_service_metadata(saved_path)
    assert svc.version == svc_metadata_new.version

    delete_saved_bento_service(svc_metadata.name, svc_metadata.version)
    delete_saved_bento_service(svc_metadata_new.name, svc_metadata_new.version)


def test_pyversion_warning_on_load(
    tmp_path_factory, capsys, example_bento_service_class
):
    # Set logging level so version mismatch warnings are outputted
    bentoml.configure_logging(logging_level=logging.WARNING)
    # (Note that logger.warning() is captured by pytest in stdout, NOT stdlog.
    #  So the warning is in capsys.readouterr().out, NOT caplog.text.)

    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack("model", test_model)

    # Should not warn for default `_python_version` value
    match_dir = tmp_path_factory.mktemp("match")
    svc.save_to_dir(match_dir)
    _ = bentoml.load(str(match_dir))
    assert "Python version mismatch" not in capsys.readouterr().out

    # Should warn for any version mismatch (major, minor, or micro)
    svc.env._python_version = "X.Y.Z"
    mismatch_dir = tmp_path_factory.mktemp("mismatch")
    svc.save_to_dir(mismatch_dir)
    _ = bentoml.load(str(mismatch_dir))
    assert "Python version mismatch" in capsys.readouterr().out

    # Reset logging level to default
    bentoml.configure_logging()
