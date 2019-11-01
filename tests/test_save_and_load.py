import os
import uuid
import mock
import pytest


import bentoml
from bentoml.handlers import DataframeHandler
from bentoml.artifact import PickleArtifact


class MyTestModel(object):
    def predict(self, input_data):
        return int(input_data) * 2


@bentoml.ver(major=2, minor=10)
@bentoml.artifacts([PickleArtifact("model")])
class MyTestBentoService(bentoml.BentoService):
    @bentoml.api(DataframeHandler)
    def predict(self, df):
        """
        An API for testing simple bento model service
        """
        return self.artifacts.model.predict(df)


@pytest.fixture()
def test_bento_service_class():
    MyTestBentoService._bento_archive_path = None
    return MyTestBentoService


def test_save_and_load_model(tmpdir, test_bento_service_class):
    test_model = MyTestModel()
    ms = test_bento_service_class.pack(model=test_model)

    assert ms.predict(1000) == 2000
    version = "test_" + uuid.uuid4().hex
    saved_path = ms.save(str(tmpdir), version=version)

    expected_version = "2.10.{}".format(version)
    assert saved_path == os.path.join(
        str(tmpdir), "MyTestBentoService", expected_version
    )
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
    test_model = MyTestModel()
    ms = test_bento_service_class()

    with mock.patch('bentoml.archive.archiver.logger') as log_mock:
        ms.save()
        log_mock.warning.assert_called_once_with(
            "Missing declared artifact '%s' for BentoService '%s'",
            'model',
            'MyTestBentoService',
        )

    ms.pack("model", test_model)
    assert ms.predict(1000) == 2000

    version = "test_" + uuid.uuid4().hex
    ms.set_version(version)

    saved_path = ms.save(str(tmpdir))

    expected_version = "2.10.{}".format(version)
    assert saved_path == os.path.join(
        str(tmpdir), "MyTestBentoService", expected_version
    )
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
    assert len(model_service.get_service_apis()) == 1
