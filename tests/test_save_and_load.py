import uuid
import mock
import pytest

from mock import patch

import bentoml
from bentoml.adapters import DataframeInput
from bentoml.saved_bundle import load_bento_service_metadata
from bentoml.exceptions import BentoMLException

from tests.conftest import delete_saved_bento_service


class TestModel(object):
    def predict(self, input_data):
        return int(input_data) * 2


def test_save_and_load_model(tmpdir, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=10)(
        example_bento_service_class
    )

    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)

    assert svc.predict(1000) == 2000
    version = "test_" + uuid.uuid4().hex

    svc.save_to_dir(str(tmpdir), version=version)
    model_service = bentoml.load(str(tmpdir))

    expected_version = "2.10.{}".format(version)
    assert model_service.version == expected_version

    api = model_service.get_inference_api('predict')
    assert api.name == "predict"
    assert api.mb_max_latency == 1000
    assert api.mb_max_batch_size == 2000
    assert isinstance(api.handler, DataframeInput)
    assert api.func(1) == 2

    # Check api methods are available
    assert model_service.predict(1) == 2


def test_warning_when_save_without_decalred_artifact(
    tmpdir, example_bento_service_class
):
    svc = example_bento_service_class()

    with mock.patch('bentoml.saved_bundle.bundler.logger') as log_mock:
        svc.save_to_dir(str(tmpdir))
        log_mock.warning.assert_called_once_with(
            "Missing declared artifact '%s' for BentoService '%s'",
            'model',
            'ExampleBentoService',
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

    api = model_service.get_inference_api('predict')
    assert api.name == "predict"
    assert isinstance(api.handler, DataframeInput)
    assert api.func(1) == 2
    # Check api methods are available
    assert model_service.predict(1) == 2


class TestBentoWithOutArtifact(bentoml.BentoService):
    __test__ = False

    @bentoml.api(input=DataframeInput())
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
        with patch.object(bentoml.BentoService, 'save_to_dir') as save_to_dir_method:
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
