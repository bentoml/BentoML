from __future__ import annotations

import re
import sys
import typing as t
import logging
import functools

import click

from bentoml.exceptions import NotFound
from bentoml._internal.utils import rich_console
from bentoml._internal.env_manager import EnvManager

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


def env_manager(func):
    from bentoml._internal.configuration.containers import BentoMLContainer

    bento_store = BentoMLContainer.bento_store.get()

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
            rich_console.print(f"loading {env} environment...")
            bento_tag = kwargs["bento"]

            # if bento_tag is a bento in the bento_store, use a persistent env
            try:
                bento = bento_store.get(bento_tag)
                env_name = str(bento.tag).replace(":", "_")
                bento_path = bento._fs.getsyspath("")
            except NotFound:
                # env created will be ephemeral
                env_name = None
                bento_path = None
            bento_env = EnvManager.get_environment(
                env_name=env_name,
                env_type=env,
                bento_path=bento_path,
            )
            rich_console.print(
                f"environment {'' if not env_name else env_name} activated!"
            )

            # once env is created, spin up a subprocess to run current arg
            bento_env.run(["bentoml"] + remove_env_arg(sys.argv[1:]))
            sys.exit(0)

        value = func(*args, **kwargs)
        return value

    return wrapper
