import os
import logging

import click
from python_on_whales import docker

from ._internal.exceptions import ManagerLoginFailed

logger = logging.getLogger(__name__)


def add_authenticate_command(cli: click.Group) -> None:
    @cli.command(name="auth-docker")
    def auth_docker():  # dead: ignore
        """
        Authenticate to Docker Hub.
        Make sure to set environment vars under .env.
        """

        user = os.environ.get("DOCKER_USER", None)
        pwd = os.environ.get("DOCKER_PASSWORD", None)
        registry = os.environ.get("DOCKER_REGISTRY", None)
        try:
            if not user or not pwd or not registry:
                logger.warning(
                    "Unable to find the given environment variables. "
                    "Make sure to put environments under .env ."
                )
                raise ManagerLoginFailed
            else:
                docker.login(username=user, password=pwd, server=registry)
        except Exception as e:  # pylint: disable=broad-except

            raise ManagerLoginFailed(f"Failed to login into (docker.io)...") from e
