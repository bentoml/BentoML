import bentoml
from bentoml.utils.usage_stats import _get_bento_service_event_properties


def test_get_bento_service_event_properties(bento_service):
    properties = _get_bento_service_event_properties(bento_service)

    assert 'PickleArtifact' in properties["artifact_types"]
    assert 'DataframeHandler' in properties["handler_types"]
    assert 'ImageHandler' in properties["handler_types"]
    assert 'JsonHandler' in properties["handler_types"]
    assert len(properties["handler_types"]) == 3

    # Disabling fastai related tests to fix travis build
    # assert 'FastaiImageHandler' in properties["handler_types"]
    # assert len(properties["handler_types"]) == 4

    assert properties["env"] is not None
    assert properties["env"]["conda_env"]["channels"] == ["defaults"]


def test_get_bento_service_event_properties_with_no_artifact():
    class TestBentoService(bentoml.BentoService):
        pass

    properties = _get_bento_service_event_properties(TestBentoService())

    assert "handler_types" not in properties
    assert properties["artifact_types"]
    assert 'NO_ARTIFACT' in properties["artifact_types"]
    assert properties["env"] is not None
