from __future__ import annotations

import os
import sys
import shlex
import shutil
import typing as t
import platform
import subprocess
from gettext import gettext

import click

conda_packages_name = "conda_packages"

_ENVVAR = [
    "BENTOML_DEBUG",
    "BENTOML_QUIET",
    "BENTOML_BUNDLE_LOCAL_BUILD",
    "BENTOML_DO_NOT_TRACK",
    "BENTOML_CONFIG",
    "BENTOML_CONFIG_OPTIONS",
    "BENTOML_PORT",
    "BENTOML_HOST",
    "BENTOML_API_WORKERS",
]
_CONDITIONAL_ENVVAR = [
    # Only print if set
    "BENTOML_NUM_THREAD",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_RS_NUM_CPUS",
    "TF_NUM_INTEROP_THREADS",
    "TF_NUM_INTRAOP_THREADS",
]


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


def pretty_format(
    info_dict: dict[str, str | list[str]], output: t.Literal["md", "plain"]
) -> str:
    out: list[str] = []
    _default_env = [f'{env}={shlex.quote(os.environ.get(env, ""))}' for env in _ENVVAR]
    _conditional_env = [
        f'{env}={shlex.quote(os.environ.get(env, ""))}'
        for env in _CONDITIONAL_ENVVAR
        if os.environ.get(env) is not None
    ]
    # environment format
    if output == "md":
        out.append(
            """\
#### Environment variable

```bash
{env}
```
""".format(
                env="\n".join(_default_env)
            )
        )
        if len(_conditional_env) > 0:
            out.extend(_conditional_env)
        out.append("#### System information\n")
        for key, value in info_dict.items():
            if isinstance(value, list):
                out.append(format_dropdown(key, value))
            else:
                out.append(f"`{key}`: {value}")
    else:
        out.extend(_default_env)
        if len(_conditional_env) > 0:
            out.extend(_conditional_env)
        for key, value in info_dict.items():
            if isinstance(value, list):
                out.append(
                    f"{key}=( " + " ".join(map(lambda s: f'"{s}"', value)) + " )"
                )
            else:
                out.append(f'{key}="{value}"')
    return "\n".join(out)


def add_env_command(cli: click.Group) -> None:
    from bentoml import __version__ as BENTOML_VERSION
    from bentoml.exceptions import CLIException
    from bentoml._internal.utils.pkg import get_pkg_version
    from bentoml._internal.utils.pkg import PackageNotFoundError

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
            info_dict["uid_gid"] = f"{os.geteuid()}:{os.getegid()}"

        if "CONDA_PREFIX" in os.environ:
            # conda packages
            conda_like = None
            for possible_exec in ["conda", "mamba", "micromamba"]:
                if shutil.which(possible_exec) is not None:
                    conda_like = possible_exec
                    break
            assert (
                conda_like is not None
            ), "couldn't find a conda-like executable, while CONDA_PREFIX is set."
            conda_packages = run_cmd([conda_like, "env", "export"])

            # user is currently in a conda environment,
            # doing this is faster than invoking `conda --version`
            try:
                conda_version = get_pkg_version("conda")
            except PackageNotFoundError:
                conda_version = run_cmd([conda_like, "--version"])[0].split(" ")[-1]

            info_dict[conda_like] = conda_version
            info_dict["in_conda_env"] = str(True)
            info_dict["conda_packages"] = conda_packages

        # process info from `pip freeze`
        pip_packages = run_cmd(["pip", "freeze"])
        info_dict["pip_packages"] = pip_packages
        click.echo(pretty_format(info_dict, output=output))
        ctx.exit(0)
