from __future__ import annotations

import os
import sys
import shutil
import typing as t
import platform
import subprocess
from gettext import gettext

import click
import psutil

from bentoml import __version__ as BENTOML_VERSION

from .yatai import add_login_command
from ..utils.pkg import get_pkg_version
from ..utils.pkg import PackageNotFoundError
from .click_utils import BentoMLCommandGroup
from .bento_server import add_serve_command
from .containerize import add_containerize_command
from .bento_management import add_bento_management_commands
from .model_management import add_model_management_commands

FC = t.TypeVar("FC", bound=t.Union[t.Callable[..., t.Any], click.Command])


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
        conda_key = "conda"

        info_dict = {
            "bentoml": BENTOML_VERSION,
            "python": platform.python_version(),
            "platform": platform.platform(),
            conda_key: "not installed",
        }

        if is_windows:
            from ctypes import windll

            # https://stackoverflow.com/a/1026626
            is_admin: bool = windll.shell32.IsUserAnAdmin() != 0
            info_dict["is_window_admin"] = str(is_admin)
        else:
            info_dict["uid:gid"] = f"{os.geteuid()}:{os.getegid()}"

        if shutil.which("conda"):
            try:
                # user is currently in a conda environment,
                # doing this is faster than invoking `conda --version`
                conda_version = get_pkg_version("conda")
            except PackageNotFoundError:
                # when user is not in a conda environment, there
                # is no way to import conda.
                conda_version = (
                    subprocess.check_output(["conda", "--version"])
                    .decode("utf-8")
                    .strip("\n")
                    .split(" ")[-1]
                )
            info_dict[conda_key] = conda_version

        click.echo("\n".join([f"{k}: {v}" for k, v in info_dict.items()]))
        ctx.exit()

    param_decls = ("--env",)

    kwargs.setdefault("is_flag", True)
    kwargs.setdefault("expose_value", False)
    kwargs.setdefault("is_eager", True)
    kwargs.setdefault("help", gettext("Print environment info and exit"))
    kwargs["callback"] = callback
    return click.option(*param_decls, **kwargs)


def create_bentoml_cli():
    from ..context import component_context

    component_context.component_name = "cli"

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
