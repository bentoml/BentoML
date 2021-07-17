import pytest
from mock import Mock, patch

import docker
from bentoml.exceptions import BentoMLException, MissingDependencyException
from bentoml.yatai.deployment.docker_utils import (
    build_docker_image,
    ensure_docker_available_or_raise,
    push_docker_image_to_repository,
)

mock_image_tag = "mytag:v1"
mock_repository = f"repository.com/{mock_image_tag}"


def test_ensure_docker_available_or_raise():
    with patch("docker.from_env") as from_env_mock:
        from_env_mock.side_effect = docker.errors.DockerException("no docker error")
        with pytest.raises(MissingDependencyException) as error:
            ensure_docker_available_or_raise()
        assert str(error.value).startswith("Docker is required")

        from_env_mock.side_effect = None
        mock_docker_client = Mock()
        mock_docker_client.ping = Mock(
            side_effect=docker.errors.APIError("no response")
        )
        from_env_mock.return_value = mock_docker_client
        with pytest.raises(MissingDependencyException) as server_error:
            ensure_docker_available_or_raise()
        assert str(server_error.value).startswith("Docker server is not responsive.")


def test_build_docker_image():
    with patch("docker.from_env") as from_env_mock:
        mock_docker_client = Mock()
        mock_docker_client.images.build = Mock(
            side_effect=docker.errors.APIError("API error")
        )
        from_env_mock.return_value = mock_docker_client
        with pytest.raises(BentoMLException) as build_api_error:
            build_docker_image(context_path="", dockerfile="", image_tag=mock_image_tag)
        assert str(build_api_error.value).startswith(
            f"Failed to build docker image {mock_image_tag}"
        )

        mock_docker_client.images.build = Mock(
            side_effect=docker.errors.BuildError(build_log="", reason="")
        )
        with pytest.raises(BentoMLException) as build_api_error:
            build_docker_image(context_path="", dockerfile="", image_tag=mock_image_tag)
        assert str(build_api_error.value).startswith(
            f"Failed to build docker image {mock_image_tag}"
        )


def test_push_docker_image_to_repository():
    with patch("docker.from_env") as from_env_mock:
        mock_docker_client = Mock()
        mock_docker_client.images.push = Mock(
            side_effect=docker.errors.APIError("API error")
        )
        from_env_mock.return_value = mock_docker_client
        with pytest.raises(BentoMLException) as push_api_error:
            push_docker_image_to_repository(mock_repository, image_tag=mock_image_tag)
        assert str(push_api_error.value).startswith(
            f"Failed to push docker image {mock_image_tag}"
        )
