import pytest

import imageio
import numpy as np

from bentoml.yatai.client import YataiClient
from tests.bento_service_examples.example_bento_service import ExampleBentoService


def pytest_configure():
    '''
    global constants for tests
    '''
    # async request client
    async def assert_request(
        method,
        url,
        headers=None,
        data=None,
        timeout=None,
        assert_status=None,
        assert_data=None,
    ):
        if assert_status is None:
            assert_status = 200

        import aiohttp

        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.request(
                    method, url, data=data, headers=headers, timeout=timeout
                ) as r:
                    data = await r.read()
        except RuntimeError:
            # the event loop has been closed due to previous task failed, ignore
            return

        if callable(assert_status):
            assert assert_status(r.status)
        else:
            assert r.status == assert_status

        if assert_data is not None:
            if callable(assert_data):
                assert assert_data(data)
            else:
                assert data == assert_data

    pytest.assert_request = assert_request

    # dataframe json orients
    pytest.DF_ORIENTS = {
        'split',
        'records',
        'index',
        'columns',
        'values',
        # 'table',  # TODO(bojiang)
    }
    pytest.DF_AUTO_ORIENTS = {
        'records',
        'columns',
    }


def pytest_addoption(parser):
    parser.addoption("--batch-request", action="store_false")


@pytest.fixture()
def is_batch_request(pytestconfig):
    return pytestconfig.getoption("batch_request")


@pytest.fixture()
def img_file(tmpdir):
    img_file_ = tmpdir.join("test_img.jpg")
    imageio.imwrite(str(img_file_), np.zeros((10, 10)))
    return str(img_file_)


@pytest.fixture()
def img_files(tmpdir):
    for i in range(10):
        img_file_ = tmpdir.join(f"test_img_{i}.jpg")
        imageio.imwrite(str(img_file_), np.zeros((10, 10)))
    return str(tmpdir.join("*.jpg"))


class TestModel(object):
    def predict_dataframe(self, df):
        return df["col1"].sum()

    def predict_image(self, input_datas):
        for input_data in input_datas:
            assert input_data is not None
        return [input_data.shape for input_data in input_datas]

    def predict_legacy_images(self, original, compared):
        return (original == compared).all()

    def predict_json(self, input_jsons):
        assert input_jsons
        return [{"ok": True}] * len(input_jsons)

    def predict_legacy_json(self, input_json):
        assert input_json is not None
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
