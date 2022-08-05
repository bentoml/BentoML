from __future__ import annotations

import os
import sys
import typing as t
import platform
import subprocess
from gettext import gettext

import click

from bentoml import __version__ as BENTOML_VERSION
from bentoml.exceptions import CLIException
from bentoml._internal.utils.pkg import get_pkg_version
from bentoml._internal.utils.pkg import PackageNotFoundError

conda_packages_name = "conda_packages"


def run_cmd(cmd: list[str]) -> list[str]:
    return subprocess.check_output(cmd).decode("utf-8").split("\n")[:-1]


def format_dropdown(title: str, content: t.Iterable[str]) -> str:
    processed = "\n".join(content)
    return f"""\
<details><summary><code>{title}</code></summary>

<br>

```{ 'yaml' if title == conda_packages_name else '' }
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


def add_env_command(cli: click.Group) -> None:
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
            raise CLIException(f"Unknown output format: {output}")

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
