import pytest
import tempfile

import bentoml
from bentoml.handlers import (
    DataframeHandler,
    ImageHandler,
    JsonHandler,
    # FastaiImageHandler,
)
from bentoml.artifact import PickleArtifact
from bentoml import config


class TestModel(object):
    def predict_dataframe(self, df):
        return df["col1"].sum()

    def predict_image(self, input_data):
        assert input_data is not None
        return input_data.shape

    def predict_json(self, input_data):
        assert input_data is not None
        return {"ok": True}


@bentoml.artifacts([PickleArtifact("model")])
@bentoml.env()
class TestBentoService(bentoml.BentoService):
    """My RestServiceTestModel packaging with BentoML
    """

    @bentoml.api(DataframeHandler, input_dtypes={"col1": "int"})
    def predict_dataframe(self, df):
        """predict_dataframe expects dataframe as input
        """
        return self.artifacts.model.predict_dataframe(df)

    @bentoml.api(ImageHandler)
    def predict_image(self, image):
        return self.artifacts.model.predict_image(image)

    @bentoml.api(ImageHandler, input_names=('original', 'compared'))
    def predict_images(self, original, compared):
        return original[0, 0] == compared[0, 0]

    @bentoml.api(JsonHandler)
    def predict_json(self, input_data):
        return self.artifacts.model.predict_json(input_data)

    # Disabling fastai related tests to fix travis build
    # @bentoml.api(FastaiImageHandler)
    # def predict_fastai_image(self, input_data):
    #     return self.artifacts.model.predict_image(input_data)
    #
    # @bentoml.api(FastaiImageHandler, input_names=('original', 'compared'))
    # def predict_fastai_images(self, original, compared):
    #     return all(original.data[0, 0] == compared.data[0, 0])


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
