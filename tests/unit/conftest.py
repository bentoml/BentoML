import functools
import glob
import inspect
import os

import imageio
import numpy as np
import pytest

# from bentoml._internal.yatai_client import YataiClient
# from tests.bento_services.service import MockBentoService


def pytest_configure():
    """
    global constants for tests
    """
    # async request client
    async def assert_request(
        method,
        url,
        headers=None,
        data=None,
        timeout=None,
        assert_status=None,
        assert_data=None,
        assert_headers=None,
    ):
        if assert_status is None:
            assert_status = 200

        import aiohttp

        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.request(
                    method, url, data=data, headers=headers, timeout=timeout
                ) as r:
                    r_body = await r.read()
        except RuntimeError:
            # the event loop has been closed due to previous task failed, ignore
            return

        if callable(assert_status):
            assert assert_status(r.status), f"{r.status} {r_body}"
        else:
            assert r.status == assert_status, f"{r.status} {r_body}"

        if assert_data is not None:
            if callable(assert_data):
                assert assert_data(r_body), r_body
            else:
                assert r_body == assert_data

        if assert_headers is not None:
            assert assert_headers(r.headers), r.headers

    pytest.assert_request = assert_request

    # dataframe json orients
    pytest.DF_ORIENTS = {
        "split",
        "records",
        "index",
        "columns",
        "values",
        # 'table',  # TODO(bojiang)
    }
    pytest.DF_AUTO_ORIENTS = {
        "records",
        "columns",
    }

    def _since_version(ver: str, skip_by_default=False):
        def _wrapper(func):
            if not inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                def _wrapped(*args, **kwargs):
                    from packaging import version

                    bundle_ver = os.environ.get("BUNDLE_BENTOML_VERSION")
                    if skip_by_default and not bundle_ver:
                        pytest.skip()
                    if bundle_ver and version.parse(bundle_ver) < version.parse(ver):
                        pytest.skip()
                    return func(*args, **kwargs)

            else:

                @functools.wraps(func)
                async def _wrapped(*args, **kwargs):
                    from packaging import version

                    bundle_ver = os.environ.get("BUNDLE_BENTOML_VERSION")
                    if skip_by_default and not bundle_ver:
                        pytest.skip()
                    if bundle_ver and version.parse(bundle_ver) < version.parse(ver):
                        pytest.skip()
                    return await func(*args, **kwargs)

            return _wrapped

        return _wrapper

    pytest.since_bentoml_version = _since_version


def pytest_addoption(parser):
    parser.addoption("--batch-request", action="store_false")


# @pytest.fixture()
# def is_batch_request(pytestconfig):
#     return pytestconfig.getoption("batch_request")
#
#
# @pytest.fixture()
# def bin_file(tmpdir):
#     bin_file_ = tmpdir.join("bin_file.bin")
#     with open(bin_file_, "wb") as of:
#         of.write("â".encode("gb18030"))
#     return str(bin_file_)
#
#
# @pytest.fixture()
# def bin_files(tmpdir):
#     for i in range(10):
#         bin_file_ = tmpdir.join(f"{i}.bin")
#         with open(bin_file_, "wb") as of:
#             of.write(f"â{i}".encode("gb18030"))
#     return sorted(glob.glob(str(tmpdir.join("*.bin"))))
#
#
# @pytest.fixture()
# def unicode_file(tmpdir):
#     bin_file_ = tmpdir.join("bin_file.unicode")
#     with open(bin_file_, "wb") as of:
#         of.write("â".encode("utf-8"))
#     return str(bin_file_)
#
#
# @pytest.fixture()
# def unicode_files(tmpdir):
#     for i in range(10):
#         bin_file_ = tmpdir.join(f"{i}.list.unicode")
#         with open(bin_file_, "wb") as of:
#             of.write(f"â{i}".encode("utf-8"))
#     return sorted(glob.glob(str(tmpdir.join("*.list.unicode"))))
#
#
# @pytest.fixture()
# def img_file(tmpdir):
#     img_file_ = tmpdir.join("test_img.jpg")
#     imageio.imwrite(str(img_file_), np.zeros((10, 10)))
#     return str(img_file_)
#
#
# @pytest.fixture()
# def img_files(tmpdir):
#     for i in range(10):
#         img_file_ = tmpdir.join(f"{i}.list.jpg")
#         imageio.imwrite(str(img_file_), np.zeros((10, 10)))
#     return sorted(glob.glob(str(tmpdir.join("*.list.jpg"))))
#
#
# @pytest.fixture()
# def json_file(tmpdir):
#     json_file_ = tmpdir.join("test.json")
#     with open(json_file_, "w") as of:
#         of.write('{"name": "kaith", "game": "morrowind"}')
#     return str(json_file_)
#
#
# @pytest.fixture()
# def json_files(tmpdir):
#     for i in range(10):
#         file_ = tmpdir.join(f"{i}.list.json")
#         with open(file_, "w") as of:
#             of.write('{"i": %d, "name": "kaith", "game": "morrowind"}' % i)
#     return sorted(glob.glob(str(tmpdir.join("*.list.json"))))
#
#
# class TestModel(object):
#     def predict_dataframe(self, df):
#         return df["col1"] * 2
#
#     def predict_image(self, input_datas):
#         for input_data in input_datas:
#             assert input_data is not None
#         return [input_data.shape for input_data in input_datas]
#
#     def predict_multi_images(self, original, compared):
#         return (original == compared).all()
#
#     def predict_json(self, input_jsons):
#         assert input_jsons
#         return [{"ok": True}] * len(input_jsons)


# @pytest.fixture()
# def example_bento_service_class():
#     # When the MockBentoService got saved and loaded again in the test, the two class
#     # attribute below got set to the loaded BentoService class. Resetting it here so it
#     # does not effect other tests
#     MockBentoService._bento_service_bundle_path = None
#     MockBentoService._bento_service_bundle_version = None
#     return MockBentoService


# @pytest.fixture()
# def bento_service(example_bento_service_class):
#     """Create a new MockBentoService
#     """
#     test_model = TestModel()
#     test_svc = example_bento_service_class()
#     test_svc.pack("model", test_model)
#     return test_svc


# @pytest.fixture()
# def bento_bundle_path(bento_service):
#     """Create a new MockBentoService, saved it to tmpdir, and return full saved_path
#     """
#     saved_path = bento_service.save()
#     yield saved_path
#     delete_saved_bento_service(bento_service.name, bento_service.version)


# def delete_saved_bento_service(name, version):
#     yc = YataiClient()
#     yc.repository.delete(f"{name}:{version}")
