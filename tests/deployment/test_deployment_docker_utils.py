import pytest

import docker
from mock import patch, Mock

from bentoml.yatai.deployment.docker_utils import ensure_docker_available_or_raise
from bentoml.exceptions import MissingDependencyException


def test_ensure_docker_available_or_raise():
    with patch('docker.from_env') as from_env_mock:
        from_env_mock.side_effect = docker.errors.DockerException('no docker error')
        with pytest.raises(MissingDependencyException) as error:
            ensure_docker_available_or_raise()
        assert str(error.value).startswith('Docker is required')

        from_env_mock.side_effect = None
        mock_docker_client = Mock()
        mock_docker_client.ping = Mock(
            side_effect=docker.errors.APIError('no response')
        )
        from_env_mock.return_value = mock_docker_client
        with pytest.raises(MissingDependencyException) as server_error:
            ensure_docker_available_or_raise()
        assert str(server_error.value).startswith('Docker server is not responsive.')
