import six
import pytest
import tempfile

import bentoml
from bentoml.handlers import (
    DataframeHandler,
    ImageHandler,
    JsonHandler,
    FastaiImageHandler,
)
from bentoml.artifact import PickleArtifact
from bentoml import config


class TestModel(object):
    def predict(self, df):
        df["age"] = df["age"].add(5)
        return df

    def predictImage(self, input_data):
        assert input_data is not None
        return [10, 24]

    def predictJson(self, input_data):
        assert input_data is not None
        return {"ok": True}

    def predictTF(self, input_data):
        assert input_data is not None
        return {"ok": True}

    def predictTorch(self, input_data):
        assert input_data is not None
        return {"ok": True}


@bentoml.artifacts([PickleArtifact("model")])
@bentoml.env()
class TestBentoService(bentoml.BentoService):
    """My RestServiceTestModel packaging with BentoML
    """

    @bentoml.api(DataframeHandler, input_dtypes={"age": "int"})
    def predict(self, df):
        """predict expects dataframe as input
        """
        return self.artifacts.model.predict(df)

    @bentoml.api(ImageHandler)
    def predictImage(self, input_data):
        return self.artifacts.model.predictImage(input_data)

    @bentoml.api(ImageHandler, input_names=('original', 'compared'))
    def predictImages(self, original, compared):
        return original[0, 0] == compared[0, 0]

    @bentoml.api(JsonHandler)
    def predictJson(self, input_data):
        return self.artifacts.model.predictJson(input_data)

    if six.PY3:

        @bentoml.api(FastaiImageHandler)
        def predictFastaiImage(self, input_data):
            return self.artifacts.model.predictImage(input_data)

        @bentoml.api(FastaiImageHandler, input_names=('original', 'compared'))
        def predictFastaiImages(self, original, compared):
            return all(original.data[0, 0] == compared.data[0, 0])


@pytest.fixture()
def bento_service():
    """Create a new TestBentoService
    """
    test_model = TestModel()

    # When the TestBentoService got saved and loaded again in the test, the two class
    # attribute below got set to the loaded BentoService class. Resetting it here so it
    # does not effect other tests
    TestBentoService._bento_service_bundle_path = None
    TestBentoService._bento_service_bundle_version = None

    test_svc = TestBentoService()
    test_svc.pack('model', test_model)
    return test_svc


@pytest.fixture()
def bento_bundle_path(bento_service, tmpdir):  # pylint:disable=redefined-outer-name
    """Create a new TestBentoService, saved it to tmpdir, and return full saved_path
    """
    saved_path = bento_service.save(str(tmpdir))
    return saved_path


@pytest.fixture(scope='session', autouse=True)
def set_test_config():
    tempdir = tempfile.mkdtemp(prefix="bentoml-test-")
    bentoml.configuration._reset_bentoml_home(tempdir)
    config().set('core', 'usage_tracking', 'false')
