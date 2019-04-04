import os
import sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import bentoml
from bentoml.model import _validate_version_str


def test_validate_version_str_fails():
    with pytest.raises(ValueError):
        _validate_version_str('44&')


def test_validate_version_str_pass():
    _validate_version_str('abc_123')

class MyFakeModel(object):
    def predict(self, input):
        return int(input) * 2

class MyFakeBentoModel(bentoml.BentoModel):
    """
    My SentimentLRModel packaging with BentoML
    """

    def config(self, artifacts, env):
        artifacts.add(bentoml.artifacts.PickleArtifact('fake_model'))

    @bentoml.api(bentoml.handlers.DataframeHandler)
    def predict(self, df):
        """
        predict expects dataframe as input
        """
        return self.artifacts.fake_model.predict(df)


BASE_TEST_PATH = "/tmp/bentoml-test"

def test_save_and_load_model():
    fake_model = MyFakeModel()
    sm = MyFakeBentoModel(fake_model=fake_model)
    assert sm.predict(1000) == 2000

    import uuid
    version = "test_" + uuid.uuid4().hex
    saved_path = sm.save(BASE_TEST_PATH, version=version)

    model_path = os.path.join(BASE_TEST_PATH, 'MyFakeBentoModel', version)
    assert os.path.exists(model_path)

    model_service = bentoml.load(saved_path, lazy_load=True)
    assert not model_service.loaded
    model_service.load()
    assert model_service.loaded

    assert len(model_service.get_service_apis()) == 1
    api = model_service.get_service_apis()[0]
    assert api.name == 'predict'
    assert api.handler == bentoml.handlers.DataframeHandler
    assert api.func(1) == 2

    # Check api methods are available
    assert model_service.predict(1) == 2

@pytest.mark.skip(reason="Setup s3 creds in travis or use a mock")
def test_save_and_load_model_from_s3():
    fake_model = MyFakeModel()
    sm = MyFakeBentoModel(fake_model=fake_model)

    s3_location = 's3://bentoml/test'
    s3_saved_path = sm.save(base_path=s3_location)
    assert s3_saved_path == os.path.join(s3_location, sm.name, sm.version)

    download_model_service = bentoml.load(s3_saved_path, lazy_load=True)
    assert not download_model_service.loaded
    download_model_service.load()
    assert download_model_service.loaded
    assert download_model_service.get_service_apis()[0].func(1) == 2
