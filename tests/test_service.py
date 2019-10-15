import sys
import pytest

import bentoml
from bentoml.handlers import DataframeHandler, FastaiImageHandler, ImageHandler
from bentoml.service import _validate_version_str


def test_custom_api_name():
    # these names should work:
    bentoml.api(DataframeHandler, api_name="a_valid_name")(lambda x: x)
    bentoml.api(DataframeHandler, api_name="AValidName")(lambda x: x)
    bentoml.api(DataframeHandler, api_name="_AValidName")(lambda x: x)
    bentoml.api(DataframeHandler, api_name="a_valid_name_123")(lambda x: x)

    with pytest.raises(ValueError) as e:
        bentoml.api(DataframeHandler, api_name="a invalid name")(lambda x: x)
    assert str(e.value).startswith("Invalid API name")

    with pytest.raises(ValueError) as e:
        bentoml.api(DataframeHandler, api_name="123_a_invalid_name")(lambda x: x)
    assert str(e.value).startswith("Invalid API name")

    with pytest.raises(ValueError) as e:
        bentoml.api(DataframeHandler, api_name="a-invalid-name")(lambda x: x)
    assert str(e.value).startswith("Invalid API name")


def test_fastai_image_handler_pip_dependencies():
    if sys.version_info < (3, 6):
        # fast ai is required 3.6 or higher.
        assert True
    else:

        class TestFastAiImageService(bentoml.BentoService):
            @bentoml.api(FastaiImageHandler)
            def test(self, image):
                return image

        service = TestFastAiImageService()

        assert 'imageio' in service._env._pip_dependencies
        assert 'fastai' in service._env._pip_dependencies


def test_image_handler_pip_dependencies():
    class TestImageService(bentoml.BentoService):
        @bentoml.api(ImageHandler)
        def test(self, image):
            return image

    service = TestImageService()
    assert 'imageio' in service._env._pip_dependencies


def test_validate_version_str_fails():
    with pytest.raises(ValueError):
        _validate_version_str("44&")


def test_validate_version_str_pass():
    _validate_version_str("abc_123")
