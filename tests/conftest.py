import pytest

import bentoml
from bentoml.artifact import PickleArtifact


class TestModel(object):
    def predict(self, df):
        df["age"] = df["age"].add(5)
        return df

    def predictImage(self, input_data):
        return [10, 24]

    def predictJson(self, input_data):
        return {"ok": True}

    def predictTF(self, input_data):
        return {"ok": True}

    def predictTorch(self, input_data):
        return {"ok": True}


@bentoml.artifacts([PickleArtifact("model")])
@bentoml.env()
class TestBentoService(bentoml.BentoService):
    """My RestServiceTestModel packaging with BentoML
    """

    @bentoml.api(bentoml.handlers.DataframeHandler, input_dtypes={"age": "int"})
    def predict(self, df):
        """predict expects dataframe as input
        """
        return self.artifacts.model.predict(df)

    @bentoml.api(bentoml.handlers.ImageHandler)
    def predictImage(self, input_data):
        return self.artifacts.model.predictImage(input_data)

    @bentoml.api(
        bentoml.handlers.ImageHandler,
        input_name=('original', 'compared'),
        accept_multiple_files=True,
    )
    def predictImages(self, original, compared):
        return original[0, 0] == compared[0, 0]

    @bentoml.api(bentoml.handlers.JsonHandler)
    def predictJson(self, input_data):
        return self.artifacts.model.predictJson(input_data)

    @bentoml.api(bentoml.handlers.TensorflowTensorHandler)
    def predictTF(self, input_data):
        return self.artifacts.model.predictTF(input_data)

    @bentoml.api(bentoml.handlers.PytorchTensorHandler)
    def predictTorch(self, input_data):
        return self.artifacts.model.predictTorch(input_data)


@pytest.fixture()
def bento_service():
    """Create a new TestBentoService
    """
    test_model = TestModel()
    return TestBentoService.pack(model=test_model)


@pytest.fixture()
def bento_archive_path(bento_service, tmpdir):  # pylint:disable=redefined-outer-name
    """Create a new TestBentoService, saved it to tmpdir, and return full saved_path
    """
    saved_path = bento_service.save(str(tmpdir))
    return saved_path


@pytest.fixture(scope='session', autouse=True)
def turn_off_tracking():
    bentoml.config.set('core', 'usage_tracking', 'false')
    return False
