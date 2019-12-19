import pytest

import bentoml
from bentoml.artifact import PickleArtifact
from bentoml.handlers import DataframeHandler, FastaiImageHandler, ImageHandler
from bentoml.service import _validate_version_str
from bentoml.exceptions import InvalidArgument


def test_custom_api_name():
    # these names should work:
    bentoml.api(DataframeHandler, api_name="a_valid_name")(lambda x: x)
    bentoml.api(DataframeHandler, api_name="AValidName")(lambda x: x)
    bentoml.api(DataframeHandler, api_name="_AValidName")(lambda x: x)
    bentoml.api(DataframeHandler, api_name="a_valid_name_123")(lambda x: x)

    with pytest.raises(InvalidArgument) as e:
        bentoml.api(DataframeHandler, api_name="a invalid name")(lambda x: x)
    assert str(e.value).startswith("Invalid API name")

    with pytest.raises(InvalidArgument) as e:
        bentoml.api(DataframeHandler, api_name="123_a_invalid_name")(lambda x: x)
    assert str(e.value).startswith("Invalid API name")

    with pytest.raises(InvalidArgument) as e:
        bentoml.api(DataframeHandler, api_name="a-invalid-name")(lambda x: x)
    assert str(e.value).startswith("Invalid API name")


@pytest.mark.skip("skip fastai tests to fix travis build")
def test_fastai_image_handler_pip_dependencies():
    class TestFastAiImageService(bentoml.BentoService):
        @bentoml.api(FastaiImageHandler)
        def test(self, image):
            return image

    service = TestFastAiImageService()

    assert 'imageio' in service._env._pip_dependencies
    assert 'fastai' in service._env._pip_dependencies


def test_invalid_artifact_type():
    with pytest.raises(InvalidArgument) as e:

        @bentoml.artifacts(["Not A Artifact"])  # pylint: disable=unused-variable
        class TestBentoService(bentoml.BentoService):
            pass

    assert "only accept list of type BentoServiceArtifact" in str(e.value)


def test_duplicated_artifact_name():
    with pytest.raises(InvalidArgument) as e:

        @bentoml.artifacts(  # pylint: disable=unused-variable
            [PickleArtifact("model"), PickleArtifact("model")]
        )
        class TestBentoService(bentoml.BentoService):
            pass

    assert "Duplicated artifact name `model` detected" in str(e.value)


def test_invalid_api_handler():
    with pytest.raises(InvalidArgument) as e:

        class TestBentoService(bentoml.BentoService):  # pylint: disable=unused-variable
            @bentoml.api("Not A BentoHandler")
            def test(self):
                pass

    assert "must be class derived from bentoml.handlers.BentoHandler" in str(e.value)


def test_image_handler_pip_dependencies():
    class TestImageService(bentoml.BentoService):
        @bentoml.api(ImageHandler)
        def test(self, image):
            return image

    service = TestImageService()
    assert 'imageio' in service._env._pip_dependencies


def test_validate_version_str_fails():
    with pytest.raises(InvalidArgument):
        _validate_version_str("44&")

    with pytest.raises(InvalidArgument):
        _validate_version_str("44 123")

    with pytest.raises(InvalidArgument):
        _validate_version_str("")


def test_validate_version_str_pass():
    _validate_version_str("abc_123")
    _validate_version_str("1")
    _validate_version_str("a_valid_version")
    _validate_version_str("AValidVersion")
    _validate_version_str("_AValidVersion")
    _validate_version_str("1.3.4")
    _validate_version_str("1.3.4-g375a71b")
