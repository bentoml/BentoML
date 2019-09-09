import os
import pytest
import uuid

import bentoml
from bentoml.artifact import PickleArtifact


class MyTestModel(object):
    def predict(self, input_data):
        return int(input_data) * 2


@bentoml.ver(major=2, minor=10)
@bentoml.env(conda_pip_dependencies=["scikit-learn"])
@bentoml.artifacts([PickleArtifact("model")])
class MyTestBentoService(bentoml.BentoService):
    @bentoml.api(bentoml.handlers.DataframeHandler)
    def predict(self, df):
        """
        An API for testing simple bento model service
        """
        return self.artifacts.model.predict(df)


def test_save_and_load_model(tmpdir):
    test_model = MyTestModel()
    ms = MyTestBentoService.pack(model=test_model)

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
    assert isinstance(api.handler, bentoml.handlers.DataframeHandler)
    assert api.func(1) == 2

    # Check api methods are available
    assert model_service.predict(1) == 2
    assert model_service.version == expected_version


class TestBentoWithOutArtifact(bentoml.BentoService):
    @bentoml.api(bentoml.handlers.DataframeHandler)
    def test(self, df):
        return df


def test_bento_without_artifact(tmpdir):
    TestBentoWithOutArtifact().save_to_dir(str(tmpdir))
    model_service = bentoml.load(str(tmpdir))
    assert len(model_service.get_service_apis()) == 1
