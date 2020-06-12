import bentoml
from bentoml.utils.usage_stats import _get_bento_service_event_properties


def test_get_bento_service_event_properties(bento_service):
    properties = _get_bento_service_event_properties(bento_service)

    assert 'PickleArtifact' in properties["artifact_types"]
    assert 'DataframeInput' in properties["input_types"]
    assert 'ImageInput' in properties["input_types"]
    assert 'LegacyImageInput' in properties["input_types"]
    assert 'JsonInput' in properties["input_types"]
    assert len(properties["input_types"]) == 4

    # Disabling fastai related tests to fix travis build
    # assert 'FastaiImageInput' in properties["input_types"]
    # assert len(properties["input_types"]) == 4

    assert properties["env"] is not None
    assert properties["env"]["conda_env"]["channels"] == ["defaults"]


def test_get_bento_service_event_properties_with_no_artifact():
    class ExampleBentoService(bentoml.BentoService):
        pass

    properties = _get_bento_service_event_properties(ExampleBentoService())

    assert "input_types" not in properties
    assert properties["artifact_types"]
    assert 'NO_ARTIFACT' in properties["artifact_types"]
    assert properties["env"] is not None
