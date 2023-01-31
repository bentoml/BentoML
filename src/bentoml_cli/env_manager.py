from __future__ import annotations

import os
import re
import sys
import typing as t
import logging
import functools

import fs
import click
from simple_di import inject
from simple_di import Provide

from bentoml.exceptions import NotFound as BentoNotFound
from bentoml.exceptions import BentoMLException
from bentoml._internal.bento.bento import Bento
from bentoml._internal.bento.bento import BentoStore
from bentoml._internal.bento.bento import BENTO_YAML_FILENAME
from bentoml._internal.bento.bento import DEFAULT_BENTO_BUILD_FILE
from bentoml._internal.env_manager import EnvManager
from bentoml._internal.configuration import get_debug_mode
from bentoml._internal.env_manager.envs import Environment
from bentoml._internal.configuration.containers import BentoMLContainer

if t.TYPE_CHECKING:
    P = t.ParamSpec("P")
    F = t.Callable[P, t.Any]

logger = logging.getLogger(__name__)


def remove_env_arg(cmd_args: list[str]) -> list[str]:
    """
    Removes `--env <env_name>` or `--env=<env_name>` from sys.argv
    """
    indices_to_remove: list[int] = []
    for i, arg in enumerate(cmd_args):
        # regex matching click.option
        if re.match(r"^--env=?.*", arg):
            indices_to_remove.append(i)
            if "=" not in arg:
                indices_to_remove.append(i + 1)

    new_cmd_args: list[str] = []
    for i, arg in enumerate(cmd_args):
        if i not in indices_to_remove:
            new_cmd_args.append(arg)
    return new_cmd_args


@inject
def get_environment(
    bento_identifier: str,
    env: str,
    bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
) -> Environment:
    # env created will be ephemeral
    if os.path.isdir(os.path.expanduser(bento_identifier)):

        bento_path_fs = fs.open_fs(
            os.path.abspath(os.path.expanduser(bento_identifier))
        )
        if bento_path_fs.isfile(BENTO_YAML_FILENAME):
            # path to a build bento dir
            return EnvManager.from_bento(
                env_type=env,
                bento=Bento.from_fs(bento_path_fs),
                is_ephemeral=True,
            ).environment
        elif bento_path_fs.isfile(DEFAULT_BENTO_BUILD_FILE):
            # path to a bento project
            raise NotImplementedError(
                "Serving bento project in an environment is not supported now."
            )
        else:
            raise BentoMLException(
                f"EnvManager failed to create environment from path {bento_path_fs}. When loading from a path, it must be either a Bento containing bento.yaml or a project directory containing bentofile.yaml"
            )
    else:
        try:
            bento = bento_store.get(bento_identifier)
            return EnvManager.from_bento(
                env_type=env,
                bento=bento,
                is_ephemeral=False,
            ).environment
        except BentoNotFound:
            # service definition
            bento_path_fs = os.path.join(os.getcwd(), DEFAULT_BENTO_BUILD_FILE)
            raise NotImplementedError(
                "Serving bento with 'import_string' in an environment is not supported now."
            )


def env_manager(func: F[t.Any]) -> F[t.Any]:
    @click.option(
        "--env",
        type=click.Choice(["conda"]),
        default=None,
        help="Environment to run the command in",
        show_default=True,
    )
    @functools.wraps(func)
    def wrapper(*args: t.Any, **kwargs: t.Any) -> t.Any:
        env = kwargs.pop("env")
        if env is not None:
            from rich.status import Status

            bento_identifier = kwargs["bento"]
            spinner_status = Status(f"Loading {env} environment")
            if not get_debug_mode():
                spinner_status.start()
                bento_env = get_environment(bento_identifier, env)
                spinner_status.stop()
            else:
                bento_env = get_environment(bento_identifier, env)
            click.echo(
                f"environment {'' if not bento_env.name else bento_env.name} activated!"
            )

            # once env is created, spin up a subprocess to run current arg
            bento_env.run(["bentoml"] + remove_env_arg(sys.argv[1:]))

        return func(*args, **kwargs)

    return wrapper
