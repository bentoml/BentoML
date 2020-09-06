import pytest

import bentoml
from bentoml.artifact import PickleArtifact
from bentoml.adapters import DataframeInput, FastaiImageInput, ImageInput
<<<<<<< HEAD
<<<<<<< HEAD
from bentoml.service import validate_version_str
=======
from bentoml.service import _validate_version_str, _validate_labels
>>>>>>> add testing for label selector
=======
from bentoml.service import _validate_version_str
>>>>>>> use enum and other updates
from bentoml.exceptions import InvalidArgument


def test_custom_api_name():
    # these names should work:
    bentoml.api(input=DataframeInput(), api_name="a_valid_name")(lambda x: x)
    bentoml.api(input=DataframeInput(), api_name="AValidName")(lambda x: x)
    bentoml.api(input=DataframeInput(), api_name="_AValidName")(lambda x: x)
    bentoml.api(input=DataframeInput(), api_name="a_valid_name_123")(lambda x: x)

    with pytest.raises(InvalidArgument) as e:
        bentoml.api(input=DataframeInput(), api_name="a invalid name")(lambda x: x)
    assert str(e.value).startswith("Invalid API name")

    with pytest.raises(InvalidArgument) as e:
        bentoml.api(input=DataframeInput(), api_name="123_a_invalid_name")(lambda x: x)
    assert str(e.value).startswith("Invalid API name")

    with pytest.raises(InvalidArgument) as e:
        bentoml.api(input=DataframeInput(), api_name="a-invalid-name")(lambda x: x)
    assert str(e.value).startswith("Invalid API name")


@pytest.mark.skip("skip fastai tests to fix CI build")
def test_fastai_image_input_pip_dependencies():
    class TestFastAiImageService(bentoml.BentoService):
        @bentoml.api(input=FastaiImageInput())
        def test(self, image):
            return image

    service = TestFastAiImageService()

    assert 'imageio' in service._env._pip_dependencies
    assert 'fastai' in service._env._pip_dependencies


# noinspection PyUnusedLocal
def test_invalid_artifact_type():
    with pytest.raises(InvalidArgument) as e:

        @bentoml.artifacts(["Not A Artifact"])
        class ExampleBentoService(  # pylint: disable=unused-variable
            bentoml.BentoService
        ):
            pass

    assert "only accept list of BentoServiceArtifact" in str(e.value)


# noinspection PyUnusedLocal
def test_duplicated_artifact_name():
    with pytest.raises(InvalidArgument) as e:

        @bentoml.artifacts([PickleArtifact("model"), PickleArtifact("model")])
        class ExampleBentoService(  # pylint: disable=unused-variable
            bentoml.BentoService
        ):
            pass

    assert "Duplicated artifact name `model` detected" in str(e.value)


# noinspection PyUnusedLocal
def test_invalid_api_input():
    with pytest.raises(InvalidArgument) as e:

        class ExampleBentoService(  # pylint: disable=unused-variable
            bentoml.BentoService
        ):
            @bentoml.api("Not A InputAdapter")
            def test(self):
                pass

    assert (
        "must be an instance of a class derived from "
        "bentoml.adapters.BaseInputAdapter" in str(e.value)
    )


def test_image_input_pip_dependencies():
    class TestImageService(bentoml.BentoService):
        @bentoml.api(input=ImageInput())
        def test(self, images):
            return images

    service = TestImageService()
    assert 'imageio' in service._env._pip_dependencies


def test_validate_version_str_fails():
    with pytest.raises(InvalidArgument):
        validate_version_str("44&")

    with pytest.raises(InvalidArgument):
        validate_version_str("44 123")

    with pytest.raises(InvalidArgument):
        validate_version_str("")


def test_validate_version_str_pass():
    _validate_version_str("abc_123")
    _validate_version_str("1")
    _validate_version_str("a_valid_version")
    _validate_version_str("AValidVersion")
    _validate_version_str("_AValidVersion")
    _validate_version_str("1.3.4")
    _validate_version_str("1.3.4-g375a71b")
