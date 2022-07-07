from __future__ import annotations

import typing as t
from gettext import gettext


import os
import sys
import platform
import click
import psutil

from bentoml import __version__ as BENTOML_VERSION
import shutil

from ..utils.pkg import get_pkg_version

from .yatai import add_login_command
from .click_utils import BentoMLCommandGroup
from .bento_server import add_serve_command
from .containerize import add_containerize_command
from .bento_management import add_bento_management_commands
from .model_management import add_model_management_commands


FC = t.TypeVar("FC", bound=t.Union[t.Callable[..., t.Any], click.Command])


def format_param(keval: tuple[str, str]) -> str:
    return "%15s : %s" % (keval[0], keval[1])


def env_option(**kwargs: t.Any) -> t.Callable[[FC], FC]:
    """Add a ``--env`` option which immediately prints environment info for debugging purposes.

    :param param_decls: One or more option names. Defaults to the single
        value ``"--help"``.
    :param kwargs: Extra arguments are passed to :func:`option`.
    """

    def callback(ctx: click.Context, params: click.Parameter, value: t.Any) -> None:
        if not value or ctx.resilient_parsing:
            return

        is_windows = sys.platform == "win32"

        info_dict = {
            "BentoML version": BENTOML_VERSION,
            "Python version": platform.python_version(),
            "Platform info": platform.platform(),
            "Conda info": "not installed",
        }

        if is_windows:
            from ctypes import windll

            # https://stackoverflow.com/a/1026626
            is_admin: bool = windll.shell32.IsUserAnAdmin() != 0
            info_dict["Is Windows admin"] = str(is_admin)
        else:
            info_dict["UID:GID"] = f"{os.geteuid()}:{os.getegid()}"

        if shutil.which("conda"):
            info_dict["Conda info"] = get_pkg_version("conda")

        click.echo("Copy-and-paste the text below in your GitHub issue.\n")
        click.echo("\n".join(map(format_param, info_dict.items())))
        ctx.exit()

    param_decls = ("--env",)

    kwargs.setdefault("is_flag", True)
    kwargs.setdefault("expose_value", False)
    kwargs.setdefault("is_eager", True)
    kwargs.setdefault("help", gettext("Print environment info and exit"))
    kwargs["callback"] = callback
    return click.option(*param_decls, **kwargs)


def create_bentoml_cli():
    CONTEXT_SETTINGS = {"help_option_names": ("-h", "--help")}

    @click.group(cls=BentoMLCommandGroup, context_settings=CONTEXT_SETTINGS)
    @click.version_option(BENTOML_VERSION, "-v", "--version")
    @env_option()
    def cli():
        """BentoML CLI"""

    # Add top-level CLI commands
    add_login_command(cli)
    add_bento_management_commands(cli)
    add_model_management_commands(cli)
    add_serve_command(cli)
    add_containerize_command(cli)

    if psutil.WINDOWS:
        import sys

        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

    return cli


cli = create_bentoml_cli()


if __name__ == "__main__":
    cli()
