import os
import sys
import typing as t
import logging
from copy import deepcopy
from typing import TYPE_CHECKING
from pathlib import Path

import fs
import click
from simple_di import inject
from simple_di import Provide
from manager._utils import shellcmd
from manager._utils import raise_exception
from manager._utils import SUPPORTED_REGISTRIES
from manager._exceptions import ManagerException
from manager._click_utils import Environment
from manager._click_utils import pass_environment
from manager._click_utils import ContainerScriptGroup
from manager._configuration import get_manifest_info
from manager._configuration import DockerManagerContainer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from manager._types import GenericDict

CONTEXT_SETTINGS = dict(auto_envvar_prefix="Authenticate")


def add_authenticate_command(cli: click.Group) -> None:
    @cli.group(
        cls=ContainerScriptGroup, name="authenticate", context_settings=CONTEXT_SETTINGS
    )
    @click.option(
        "--registry",
        required=False,
        type=click.Choice(SUPPORTED_REGISTRIES),
        multiple=True,
        help="Targets registry to login.",
    )
    @pass_environment
    @inject
    def authenticate_cli(
        ctx: Environment,
        registry: t.Optional[t.Iterable[str]],
        default_context: "t.Tuple[GenericDict]" = Provide[
            DockerManagerContainer.default_context
        ],
    ):
        """Authenticate to multiple registries. (ECR, GCR, Docker.io)"""

        from dotenv import load_dotenv

        _ = load_dotenv(dotenv_path=ctx._fs.getsyspath(".env"))

        if ctx.docker_package != DockerManagerContainer.default_name:
            _, loaded_registries = get_manifest_info(
                docker_package=ctx.docker_package,
                cuda_version=ctx.cuda_version,
            )
        else:
            _, loaded_registries = default_context  # type: ignore

        if registry:
            registries = {k: loaded_registries[k] for k in registry}
        else:
            registries = deepcopy(loaded_registries)

        ctx.registries = registries

    @cli.command(name="push-readmes")
    @inject
    @raise_exception
    @pass_environment
    def push_readmes(ctx: Environment) -> None:
        """
        Push a docker package's README to registered hubs.

        \b
        Users need to install docker-pushrm at $HOME/.docker/cli-plugins in order to use this command.
        Usually this is not needed since we setup a CI for this :). However, this is more convenient.
        """
        # sanity check
        if not Path(
            os.path.expandvars("$HOME/.docker/cli-plugins/"), "docker-pushrm"
        ).exists():
            logger.error(
                "docker-pushrm is not found. Hence, this cmd can't be used. Exitting..."
            )
            sys.exit(1)

        for repo in ctx.registries.values():
            click_ctx = click.get_current_context()
            authenticate_cmd_group = t.cast(
                ContainerScriptGroup, cli.commands["authenticate"]
            )
            containerscript_cli: t.Optional[
                click.Command
            ] = authenticate_cmd_group.get_command(click_ctx, repo.provider)
            if containerscript_cli is None:
                raise ManagerException(
                    f"{repo.provider} is not yet supported under containerscript/"
                )

            readmes = fs.path.combine(ctx.docker_package, "README.md")
            url = os.environ.get(repo.url, None)
            if url is not None and repo.provider in ["dockerhub", "quay", "harborv2"]:
                click_ctx.invoke(containerscript_cli)
                cmd_args = [
                    "pushrm",
                    "--file",
                    ctx._generated_dir.getsyspath(readmes),
                    "--provider",
                    repo.provider,
                    url,
                ]
                _ = shellcmd("docker", *cmd_args)
