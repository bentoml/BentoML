from __future__ import annotations

import os
import re
import sys
import typing as t
import logging
import functools
from shutil import which

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
    env: t.Literal["conda"],
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
                env_type=env, bento=Bento.from_fs(bento_path_fs)
            ).environment
        elif bento_path_fs.isfile(DEFAULT_BENTO_BUILD_FILE):
            # path to a bento project
            raise NotImplementedError(
                "Serving from development project is not yet supported."
            )
        else:
            raise BentoMLException(
                f"EnvManager failed to create an environment from path {bento_path_fs}. When loading from a path, it must be either a Bento or a project directory containing '{DEFAULT_BENTO_BUILD_FILE}'."
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
            raise NotImplementedError(
                "Serving bento with 'import_string' in an environment is not supported now."
            )


def env_manager(func: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
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
            spinner_status = Status(f"Preparing {env} environment")
            if not get_debug_mode():
                spinner_status.start()
                bento_env = get_environment(bento_identifier, env)
                spinner_status.stop()
            else:
                bento_env = get_environment(bento_identifier, env)

            # once env is created, spin up a subprocess to run current arg
            bentoml_exec_path = which("bentoml")
            if bentoml_exec_path is None:
                raise BentoMLException("bentoml command not found!")
            bento_env.run([bentoml_exec_path] + remove_env_arg(sys.argv[1:]))

        return func(*args, **kwargs)

    return wrapper
