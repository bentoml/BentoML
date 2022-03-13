import os
import logging

import click
from manager._utils import shellcmd
from python_on_whales import docker
from manager._exceptions import ManagerLoginFailed
from manager._click_utils import Environment
from manager._click_utils import pass_environment

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
                f"Unable to find the given environment variables. "
                "Make sure to put environments under .env ."
            )
            raise ManagerLoginFailed
        else:
            docker.login(username=user, password=pwd)
    except Exception as e:  # pylint: disable=broad-except
        raise ManagerLoginFailed(
            f"Failed to login into {docker_info.url} (docker.io) as {docker_info.user}..."
        ) from e
