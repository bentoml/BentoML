import os
import logging

import click
from python_on_whales import docker

from ..groups import Environment
from ..groups import pass_environment
from ..exceptions import ManagerLoginFailed

logger = logging.getLogger(__name__)


@click.command(name="login-docker-io")
@pass_environment
def main(ctx: Environment, *args, **kwargs) -> None:
    """
    Authenticate to Docker Hub.
    Make sure to set environment vars under .env.
    """

    docker_info = ctx.registries["docker.io"]
    assert docker_info.password_type == "envars"

    user = os.environ.get(docker_info.user, None)
    pwd = os.environ.get(docker_info.password, None)
    url = os.environ.get(docker_info.url, None)
    try:
        if not user or not pwd or not url:
            logger.warning(
                "Unable to find the given environment variables. "
                "Make sure to put environments under .env ."
            )
            raise ManagerLoginFailed
        else:
            docker.login(username=user, password=pwd, server=url)
    except Exception as e:  # pylint: disable=broad-except
        raise ManagerLoginFailed(
            f"Failed to login into {docker_info.url} (docker.io) as {docker_info.user}..."
        ) from e
