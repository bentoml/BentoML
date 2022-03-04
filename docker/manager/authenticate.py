import os
import sys
import typing as t
import logging
from typing import TYPE_CHECKING
from pathlib import Path

import click
from simple_di import inject
from simple_di import Provide
from manager._utils import run
from manager._utils import as_posix
from manager._utils import shellcmd
from manager._utils import raise_exception
from manager._container import ManagerContainer
from manager._container import get_registry_context
from manager.exceptions import ManagerException
from manager.exceptions import ManagerLoginFailed

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from manager._container import RegistryCtx


def add_authenticate_command(cli: click.Group) -> None:
    @cli.command()
    @click.option(
        "--registry",
        required=False,
        type=click.STRING,
        default=["docker.io"],
        multiple=True,
        help="Targeted registry to login.",
    )
    @click.option(
        "--users",
        required=False,
        type=click.STRING,
        help="Users to login that overwrite with default config [Optional].",
    )
    @inject
    def authenticate(
        docker_package: str,
        registry: t.Optional[t.Tuple[str]],
        users: t.Optional[str],
        default_registries: "t.List[RegistryCtx]" = Provide[
            ManagerContainer.bento_server_registry_ctx
        ],
    ) -> None:
        """
        Authenticate to a given Docker registry. Currently supports docker.io, quay.io and Habor V2.
        By default we will log into docker.io

        \b
        Usage:
            manager authenticate bento-server --registry quay.io

        """
        from dotenv import load_dotenv

        _ = load_dotenv(
            dotenv_path=ManagerContainer.docker_dir.joinpath(".env").as_posix()
        )
        if docker_package != ManagerContainer.bento_server_name:
            registries = get_registry_context(package=docker_package)
        else:
            registries = default_registries

        if registry not in [i.name for i in registries]:
            raise ManagerException(f"registry is not found under {docker_package}.yaml")
        for r in registries:
            try:
                user = os.environ.get(r.user, "")
                pwd = os.environ.get(r.password, "")
                if not user or not pwd:
                    logger.warning(
                        "Unable to find the given environment variables. Make sure to setup registry correctly. "
                        f"If setup a local `.env` under {ManagerContainer.docker_dir} then safely ignore this warning message."
                    )
                _ = shellcmd(
                    "docker",
                    "login",
                    r.url,
                    "--user",
                    user,
                    "--password",
                    pwd,
                    shell=True,
                )
            except Exception as e:  # pylint: disable=broad-except
                if "docker.io" in r.name:
                    logger.warning(
                        "Failed to login with scripts. Try `docker login` interactively to log into Docker Hub."
                    )
                raise ManagerLoginFailed(
                    f"Failed to login into {r.url} ({r.name}) as {r.user}..."
                ) from e

    @cli.command()
    @click.option(
        "--users",
        required=False,
        type=click.STRING,
        help="Users to login that overwrite with default config [Optional].",
    )
    @raise_exception
    @inject
    def push_readmes(
        docker_package: str,
        users: t.Optional[str],
        default_registries: "t.List[RegistryCtx]" = Provide[
            ManagerContainer.bento_server_registry_ctx
        ],
    ) -> None:
        """
        Push a docker package's README to registered hubs.

        \b
        Users need to install docker-pushrm at $HOME/.docker/cli-plugins in order to use this command.
        """
        # sanity check
        if not Path(
            os.path.expandvars("$HOME/.docker/cli-plugins/"), "docker-pushrm"
        ).exists():
            logger.error(
                "docker-pushrm is not found. Hence, this cmd can't be used. Exitting..."
            )
            sys.exit(1)

        if docker_package != ManagerContainer.bento_server_name:
            registries = get_registry_context(package=docker_package)
        else:
            registries = default_registries

        for repo in registries:
            run(
                "docker",
                "pushrm",
                "--file",
                as_posix(os.getcwd(), "generated", repo.name, "README.md"),
                "--provider",
                repo.provider,
                repo.url,
            )
