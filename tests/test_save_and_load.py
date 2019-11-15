import os
import uuid
import mock
import pytest

from mock import patch

import bentoml
from bentoml.handlers import DataframeHandler
from bentoml.artifact import PickleArtifact
from bentoml.bundler import load_bento_service_metadata
from bentoml.exceptions import BentoMLException


class TestModel(object):
    def predict(self, input_data):
        return int(input_data) * 2


@bentoml.ver(major=2, minor=10)
@bentoml.artifacts([PickleArtifact("model")])
class TestBentoService(bentoml.BentoService):
    @bentoml.api(DataframeHandler)
    def predict(self, df):
        """
        An API for testing simple bento model service
        """
        return self.artifacts.model.predict(df)


# pylint: disable=redefined-outer-name
@pytest.fixture()
def test_bento_service_class():
    # When the TestBentoService got saved and loaded again in the test, the two class
    # attribute below got set to the loaded BentoService class. Resetting it here so it
    # does not effect other tests
    TestBentoService._bento_service_bundle_path = None
    TestBentoService._bento_service_bundle_version = None
    return TestBentoService


def test_save_and_load_model(tmpdir, test_bento_service_class):
    test_model = TestModel()
    svc = test_bento_service_class.pack(model=test_model)

    assert svc.predict(1000) == 2000
    version = "test_" + uuid.uuid4().hex
    saved_path = svc.save(str(tmpdir), version=version)

    expected_version = "2.10.{}".format(version)
    assert saved_path == os.path.join(str(tmpdir), "TestBentoService", expected_version)
    assert os.path.exists(saved_path)

    model_service = bentoml.load(saved_path)

    assert len(model_service.get_service_apis()) == 1
    api = model_service.get_service_apis()[0]
    assert api.name == "predict"
    assert isinstance(api.handler, DataframeHandler)
    assert api.func(1) == 2

    # Check api methods are available
    assert model_service.predict(1) == 2
    assert model_service.version == expected_version


def test_pack_on_bento_service_instance(tmpdir, test_bento_service_class):
    test_model = TestModel()
    svc = test_bento_service_class()

    with mock.patch('bentoml.bundler.bundler.logger') as log_mock:
        svc.save()
        log_mock.warning.assert_called_once_with(
            "Missing declared artifact '%s' for BentoService '%s'",
            'model',
            'TestBentoService',
        )

    svc.pack("model", test_model)
    assert svc.predict(1000) == 2000

    version = "test_" + uuid.uuid4().hex
    svc.set_version(version)

    saved_path = svc.save(str(tmpdir))

    expected_version = "2.10.{}".format(version)
    assert saved_path == os.path.join(str(tmpdir), "TestBentoService", expected_version)
    assert os.path.exists(saved_path)

    model_service = bentoml.load(saved_path)

    assert len(model_service.get_service_apis()) == 1
    api = model_service.get_service_apis()[0]
    assert api.name == "predict"
    assert isinstance(api.handler, DataframeHandler)
    assert api.func(1) == 2

    # Check api methods are available
    assert model_service.predict(1) == 2
    assert model_service.version == expected_version


class TestBentoWithOutArtifact(bentoml.BentoService):
    @bentoml.api(DataframeHandler)
    def test(self, df):
        return df


def test_bento_without_artifact(tmpdir):
    TestBentoWithOutArtifact().save_to_dir(str(tmpdir))
    model_service = bentoml.load(str(tmpdir))
    assert model_service.test(1) == 1
    assert len(model_service.get_service_apis()) == 1


def test_save_duplicated_bento_exception_raised(tmpdir, test_bento_service_class):
    test_model = TestModel()
    svc = test_bento_service_class()
    svc.pack("model", test_model)

    saved_path = svc.save(str(tmpdir))
    svc_metadata = load_bento_service_metadata(saved_path)
    assert svc.version == svc_metadata.version

    with pytest.raises(BentoMLException):
        with patch.object(bentoml.BentoService, 'save_to_dir') as save_to_dir_method:
            # attempt to save again
            svc.save(str(tmpdir))
            save_to_dir_method.assert_not_called()

    # reset svc version
    svc.set_version()
    saved_path = svc.save(str(tmpdir))
    svc_metadata = load_bento_service_metadata(saved_path)
    assert svc.version == svc_metadata.version
