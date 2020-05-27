import pytest

import imageio
import numpy as np

from bentoml.yatai.client import YataiClient
from tests.bento_service_examples.example_bento_service import ExampleBentoService


@pytest.fixture()
def img_file(tmpdir):
    img_file_ = tmpdir.join("test_img.jpg")
    imageio.imwrite(str(img_file_), np.zeros((10, 10)))
    return img_file_


class TestModel(object):
    def predict_dataframe(self, df):
        return df["col1"].sum()

    def predict_image(self, input_datas):
        for input_data in input_datas:
            assert input_data is not None
        return [input_data.shape for input_data in input_datas]

    def predict_json(self, input_data):
        assert input_data is not None
        return {"ok": True}


@pytest.fixture()
def example_bento_service_class():
    # When the ExampleBentoService got saved and loaded again in the test, the two class
    # attribute below got set to the loaded BentoService class. Resetting it here so it
    # does not effect other tests
    ExampleBentoService._bento_service_bundle_path = None
    ExampleBentoService._bento_service_bundle_version = None
    return ExampleBentoService


@pytest.fixture()
def bento_service(example_bento_service_class):  # pylint:disable=redefined-outer-name
    """Create a new ExampleBentoService
    """
    test_model = TestModel()
    test_svc = example_bento_service_class()
    test_svc.pack('model', test_model)
    return test_svc


@pytest.fixture()
def bento_bundle_path(bento_service):  # pylint:disable=redefined-outer-name
    """Create a new ExampleBentoService, saved it to tmpdir, and return full saved_path
    """
    saved_path = bento_service.save()
    yield saved_path
    delete_saved_bento_service(bento_service.name, bento_service.version)


def delete_saved_bento_service(name, version):
    yc = YataiClient()
    yc.repository.dangerously_delete_bento(name, version)
