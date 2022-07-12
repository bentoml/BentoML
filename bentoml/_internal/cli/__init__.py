from __future__ import annotations

import os
import typing as t
import platform
import subprocess
from gettext import gettext

import click
import psutil

from bentoml import __version__ as BENTOML_VERSION
from bentoml.exceptions import BentoMLException

from .yatai import add_login_command
from ..utils.pkg import get_pkg_version
from ..utils.pkg import PackageNotFoundError
from .click_utils import BentoMLCommandGroup
from .bento_server import add_serve_command
from .containerize import add_containerize_command
from .bento_management import add_bento_management_commands
from .model_management import add_model_management_commands

conda_packages_name = "conda_packages"


def run_cmd(cmd: list[str]) -> list[str]:
    return subprocess.check_output(cmd).decode("utf-8").split("\n")[:-1]


def format_dropdown(title: str, content: t.Iterable[str]) -> str:
    processed = "\n".join(content)
    return f"""\
<details><summary><code>{title}</code></summary>

<br>

```{ 'yaml' if title == conda_packages_name else 'markdown' }
{processed}
```

</details>
"""


def format_keys(key: str, markdown: bool) -> str:
    return f"`{key}`" if markdown else key


def pretty_format(
    info_dict: dict[str, str | list[str]], output: t.Literal["md", "plain"]
) -> str:
    out: list[str] = []
    for key, value in info_dict.items():
        if isinstance(value, list):
            if output == "md":
                out.append(format_dropdown(key, value))
            else:
                out.append(f"{key}:\n  " + "\n  ".join(value))
        else:
            out.append(f"{format_keys(key, markdown=output=='md')}: {value}")
    return "\n".join(out)


def create_bentoml_cli():

    import sys

    from ..context import component_context

    component_context.component_name = "cli"

    CONTEXT_SETTINGS = {"help_option_names": ("-h", "--help")}

    @click.group(cls=BentoMLCommandGroup, context_settings=CONTEXT_SETTINGS)
    @click.version_option(BENTOML_VERSION, "-v", "--version")
    def cli():
        """BentoML CLI"""

    @cli.command(help=gettext("Print environment info and exit"))
    @click.option(
        "-o",
        "--output",
        type=click.Choice(["md", "plain"]),
        default="md",
        show_default=True,
        help="Output format. '-o plain' to display without format.",
    )
    @click.pass_context
    def env(ctx: click.Context, output: t.Literal["md", "plain"]) -> None:  # type: ignore (unused warning)
        if output not in ["md", "plain"]:
            raise BentoMLException(f"Unknown output format: {output}")

        is_windows = sys.platform == "win32"

        info_dict: dict[str, str | list[str]] = {
            "bentoml": BENTOML_VERSION,
            "python": platform.python_version(),
            "platform": platform.platform(),
        }

        if is_windows:
            from ctypes import windll

            # https://stackoverflow.com/a/1026626
            is_admin: bool = windll.shell32.IsUserAnAdmin() != 0
            info_dict["is_window_admin"] = str(is_admin)
        else:
            info_dict["uid:gid"] = f"{os.geteuid()}:{os.getegid()}"

        if "CONDA_PREFIX" in os.environ:
            # conda packages
            conda_packages = run_cmd(["conda", "env", "export"])

            # user is currently in a conda environment,
            # doing this is faster than invoking `conda --version`
            try:
                conda_version = get_pkg_version("conda")
            except PackageNotFoundError:
                conda_version = run_cmd(["conda", "--version"])[0].split(" ")[-1]

            info_dict["conda"] = conda_version
            info_dict["in_conda_env"] = str(True)
            info_dict["conda_packages"] = conda_packages
        else:
            # process info from `pip freeze`
            pip_packages = run_cmd(["pip", "freeze"])
            info_dict["pip_packages"] = pip_packages
        click.echo(pretty_format(info_dict, output=output))
        ctx.exit(0)

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
