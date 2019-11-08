import six
from bentoml.utils.usage_stats import _get_bento_service_event_properties

def test_get_bento_service_event_properties(bento_service):

    properties = _get_bento_service_event_properties(bento_service)

    assert 'PickleArtifact' in properties["artifact_types"]

    assert 'DataframeHandler' in properties["handler_types"]
    assert 'ImageHandler' in properties["handler_types"]
    assert 'JsonHandler' in properties["handler_types"]

    if six.PY3:
        assert 'FastaiImageHandler' in properties["handler_types"]
        assert len(properties["handler_types"]) == 4
    else:
        assert len(properties["handler_types"]) == 3

    assert properties["env"] is not None