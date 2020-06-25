import json

from click.testing import CliRunner
from mock import MagicMock, patch

import bentoml
from bentoml.cli import create_bentoml_cli
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


def mock_track_cli(event, properties):
    print(event)
    print(json.dumps(properties))


@patch('bentoml.cli.click_utils.track_cli', MagicMock(side_effect=mock_track_cli))
def test_track_cli_usage(bento_service, bento_bundle_path):
    runner = CliRunner()
    cli = create_bentoml_cli()
    successful_result = runner.invoke(
        cli.commands['info'], [f'{bento_service.name}:{bento_service.version}']
    )
    track_result = successful_result.output.strip().split('\n')[-2:]
    assert successful_result.exit_code == 0
    assert track_result[0] == 'info'
    additional_info = json.loads(track_result[1])
    assert additional_info['duration']


@patch('bentoml.cli.click_utils.track_cli', MagicMock(side_effect=mock_track_cli))
def test_track_cli_with_click_exception():
    runner = CliRunner()
    cli = create_bentoml_cli()
    failed_result = runner.invoke(
        cli.commands['azure-functions'], ['update', 'mock-deployment-name']
    )
    track_result = failed_result.output.strip().split('\n')
    print(track_result)
    assert failed_result.exit_code == 1
    event_name = track_result[0].split('\r')[1]
    assert event_name == 'deploy-update'
    additional_info = json.loads(track_result[1])
    assert additional_info['platform'] == 'AZURE_FUNCTIONS'


def mock_bento_api_server(bento_service, port):
    raise KeyboardInterrupt()


@patch('bentoml.cli.click_utils.track_cli', MagicMock(side_effect=mock_track_cli))
@patch('bentoml.cli.BentoAPIServer', MagicMock(side_effect=mock_bento_api_server))
def test_track_cli_with_keyboard_interrupt(bento_bundle_path):
    runner = CliRunner()
    cli = create_bentoml_cli()

    failed_result = runner.invoke(cli.commands['serve'], [bento_bundle_path])
    track_result = failed_result.output.strip().split('\n')[-4:]
    assert failed_result.exit_code == 1
    assert track_result[0] == 'serve'
    additional_info = json.loads(track_result[1])
    assert additional_info['error_type'] == 'KeyboardInterrupt'
