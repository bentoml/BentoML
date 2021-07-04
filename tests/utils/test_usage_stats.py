from os import environ
from click.testing import CliRunner
from mock import MagicMock, patch

import bentoml
from bentoml.cli import create_bentoml_cli
from bentoml.utils.usage_stats import _get_bento_service_event_properties, _do_not_track
from bentoml.yatai.proto.deployment_pb2 import DeleteDeploymentResponse, Deployment
from bentoml.yatai.status import Status

MOCK_DEPLOYMENT_NAME = 'mock_name'
MOCK_FAILED_DEPLOYMENT_NAME = 'mock-failed'
MOCK_DEPLOYMENT_NAMESPACE = 'mock_namespace'

DO_NOT_TRACK_ENV = "BENTOML_DO_NOT_TRACK"


def mock_track_func(event, properties):
    return event, properties


def mock_deployment_pb(name=MOCK_DEPLOYMENT_NAME):
    mock_deployment = Deployment(name=name, namespace='mock-namespace')
    mock_deployment.spec.azure_functions_operator_config.location = 'mock-location'
    mock_deployment.spec.operator = 4
    mock_deployment.created_at.GetCurrentTime()
    return mock_deployment


def mock_delete_deployment(deployment_pb):
    if deployment_pb.name == MOCK_FAILED_DEPLOYMENT_NAME:
        return DeleteDeploymentResponse(status=Status.ABORTED())
    else:
        return DeleteDeploymentResponse(status=Status.OK())


def mock_get_operator_func():
    def func(yatai_service, deployment_pb):
        operator = MagicMock()
        operator.delete.side_effect = mock_delete_deployment
        return operator

    return func


def mock_start_dev_server(
    bundle_path: str,
    port: int = 5000,
    mb_max_latency: int = 0,
    mb_max_batch_size: int = 0,
    run_with_ngrok: bool = False,
    enable_swagger: bool = False,
    config_file: str = None,
):
    raise KeyboardInterrupt()


def test_get_bento_service_event_properties(bento_service):
    properties = _get_bento_service_event_properties(bento_service)

    assert 'PickleArtifact' in properties["artifact_types"]
    assert 'DataframeInput' in properties["input_types"]
    assert 'ImageInput' in properties["input_types"]
    assert 'JsonInput' in properties["input_types"]
    assert len(properties["input_types"]) == 4

    assert properties["env"] is not None


def test_get_bento_service_event_properties_with_no_artifact():
    class ExampleBentoService(bentoml.BentoService):
        pass

    properties = _get_bento_service_event_properties(ExampleBentoService())

    assert "input_types" not in properties
    assert properties["artifact_types"]
    assert 'NO_ARTIFACT' in properties["artifact_types"]
    assert properties["env"] is not None


def test_track_cli_usage(bento_service, bento_bundle_path):
    with patch('bentoml.cli.click_utils.track') as mock:
        mock.side_effect = mock_track_func
        runner = CliRunner()
        cli = create_bentoml_cli()
        runner.invoke(
            cli.commands['info'], [f'{bento_service.name}:{bento_service.version}']
        )
        event_name, properties = mock.call_args_list[0][0]
        assert event_name == 'bentoml-cli'
        assert properties['command'] == 'info'
        assert properties['return_code'] == 0
        assert properties['duration']


def test_track_cli_with_keyboard_interrupt(bento_bundle_path):
    with patch('bentoml.cli.click_utils.track') as track:
        track.side_effect = mock_track_func
        with patch('bentoml.cli.bento_service.start_dev_server') as start_dev_server:
            start_dev_server.side_effect = mock_start_dev_server
            runner = CliRunner()
            cli = create_bentoml_cli()
            runner.invoke(cli.commands['serve'], [bento_bundle_path])
            _, properties = track.call_args_list[0][0]
            assert properties['return_code'] == 2
            assert properties['error_type'] == 'KeyboardInterrupt'
            assert properties['command'] == 'serve'


def test_do_not_track():
    """Test _do_not_track behavior with different environment variable values.

    If BENTOML_DO_NOT_TRACK is not set, False should be returned. Else if
    BENTOML_DO_NOT_TRACK is True, True should be returned. Else if
    BENTOML_DO_NOT_TRACK is False, False should be returned. Calling _do_not_track()
    multiple times will return the same value.
    """
    _do_not_track.cache_clear()
    if DO_NOT_TRACK_ENV in environ:
        del environ[DO_NOT_TRACK_ENV]
    assert not _do_not_track()

    _do_not_track.cache_clear()
    environ[DO_NOT_TRACK_ENV] = str(True)
    assert _do_not_track()

    _do_not_track.cache_clear()
    environ[DO_NOT_TRACK_ENV] = str(False)
    assert not _do_not_track()

    environ[DO_NOT_TRACK_ENV] = str(True)
    assert not _do_not_track()
