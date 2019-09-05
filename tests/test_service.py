import pytest

import bentoml
from bentoml.service import _validate_version_str


def test_custom_api_name():
    # these names should work:
    bentoml.api(bentoml.handlers.DataframeHandler, api_name="a_valid_name")(lambda x: x)
    bentoml.api(bentoml.handlers.DataframeHandler, api_name="AValidName")(lambda x: x)
    bentoml.api(bentoml.handlers.DataframeHandler, api_name="_AValidName")(lambda x: x)
    bentoml.api(bentoml.handlers.DataframeHandler, api_name="a_valid_name_123")(
        lambda x: x
    )

    with pytest.raises(ValueError) as e:
        bentoml.api(bentoml.handlers.DataframeHandler, api_name="a invalid name")(
            lambda x: x
        )
    assert str(e.value).startswith("Invalid API name")

    with pytest.raises(ValueError) as e:
        bentoml.api(bentoml.handlers.DataframeHandler, api_name="123_a_invalid_name")(
            lambda x: x
        )
    assert str(e.value).startswith("Invalid API name")

    with pytest.raises(ValueError) as e:
        bentoml.api(bentoml.handlers.DataframeHandler, api_name="a-invalid-name")(
            lambda x: x
        )
    assert str(e.value).startswith("Invalid API name")


def test_handler_pip_dependencies():
    @bentoml.artifacts([bentoml.artifact.PickleArtifact('artifact')])
    class TestModel(bentoml.BentoService):
        @bentoml.api(bentoml.handlers.FastaiImageHandler)
        def test(self, image):
            return image

    empy_artifact = []
    service = TestModel.pack(artifact=empy_artifact)

    assert 'imageio' in service._env._pip_dependencies
    assert 'fastai' in service._env._pip_dependencies


def test_validate_version_str_fails():
    with pytest.raises(ValueError):
        _validate_version_str("44&")

def test_validate_version_str_pass():
    _validate_version_str("abc_123")
