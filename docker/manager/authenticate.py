import typing as t
import logging
from copy import deepcopy
from typing import TYPE_CHECKING

import click
from simple_di import inject
from simple_di import Provide

from ._internal.utils import SUPPORTED_REGISTRIES
from ._internal.groups import Environment
from ._internal.groups import pass_environment
from ._internal.groups import ContainerScriptGroup
from ._internal.configuration import get_manifest_info
from ._internal.configuration import DockerManagerContainer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ._internal.types import GenericDict

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
