import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import bentoml  # noqa: E402
from bentoml.artifact import PickleArtifact  # noqa: E402


class MyTestModel(object):

    def predict(self, input_data):
        return int(input_data) * 2


@bentoml.env(conda_pip_dependencies=['scikit-learn'])
@bentoml.artifacts([PickleArtifact('model')])
class MyTestBentoService(bentoml.BentoService):

    @bentoml.api(bentoml.handlers.DataframeHandler)
    def predict(self, df):
        """
        An API for testing simple bento model service
        """
        return self.artifacts.model.predict(df)


BASE_TEST_PATH = "/tmp/bentoml-test"


def test_save_and_load_model():
    test_model = MyTestModel()
    ms = MyTestBentoService.pack(model=test_model)

    assert ms.predict(1000) == 2000

    import uuid
    version = "test_" + uuid.uuid4().hex
    saved_path = ms.save(BASE_TEST_PATH, version=version)

    model_path = os.path.join(BASE_TEST_PATH, 'MyTestBentoService', version)
    assert os.path.exists(model_path)

    model_service = bentoml.load(saved_path)

    assert len(model_service.get_service_apis()) == 1
    api = model_service.get_service_apis()[0]
    assert api.name == 'predict'
    assert isinstance(api.handler, bentoml.handlers.DataframeHandler)
    assert api.func(1) == 2

    # Check api methods are available
    assert model_service.predict(1) == 2

    assert model_service.version == version


@pytest.mark.skip(reason="Setup s3 creds in travis or use a mock")
def test_save_and_load_model_from_s3():
    test_model = MyTestModel()
    ms = MyTestBentoService.pack(model=test_model)

    s3_location = 's3://bentoml/test'
    s3_saved_path = ms.save(base_path=s3_location)

    download_model_service = bentoml.load(s3_saved_path)
    assert download_model_service.get_service_apis()[0].func(1) == 2
