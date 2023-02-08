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
    "BENTOML_RETRY_RUNNER_REQUESTS",
    "BENTOML_CONTAINERIZE_BACKEND",
    "BENTOML_NUM_THREAD",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_RS_NUM_CPUS",
    "TF_NUM_INTEROP_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "CUDA_VISIBLE_DEVICES",
]


def run_cmd(cmd: list[str]) -> list[str]:
    return subprocess.check_output(cmd).decode("utf-8").split("\n")[:-1]


def _format_dropdown(title: str, content: t.Iterable[str]) -> str:
    processed = "\n".join(content)
    return f"""\
<details><summary><code>{title}</code></summary>

<br>

```{ 'yaml' if title == conda_packages_name else '' }
{processed}
```

</details>
"""


def _format_env(env: list[str]) -> list[str]:
    return [f"{e}={shlex.quote(os.environ.get(e, ''))}" for e in env]


def format_md(env: list[str], info_dict: dict[str, str | list[str]]) -> list[str]:
    out: list[str] = []
    out.append(
        """\
#### Environment variable

```bash
{env}
```
""".format(
            env="\n".join(_format_env(env))
        )
    )
    out.append("#### System information\n")
    for key, value in info_dict.items():
        if isinstance(value, list):
            out.append(_format_dropdown(key, value))
        else:
            out.append(f"`{key}`: {value}")
    return out


def format_bash(env: list[str], info_dict: dict[str, str | list[str]]) -> list[str]:
    out: list[str] = []
    out.extend(_format_env(env))
    for key, value in info_dict.items():
        if isinstance(value, list):
            out.append(f"{key}=( " + " ".join(map(lambda s: f'"{s}"', value)) + " )")
        else:
            out.append(f'{key}="{value}"')
    return out


def pretty_format(
    info_dict: dict[str, str | list[str]], output: t.Literal["md", "bash"]
) -> str:
    env = _ENVVAR + list(filter(lambda e: e in os.environ, _CONDITIONAL_ENVVAR))
    return "\n".join({"md": format_md, "bash": format_bash}[output](env, info_dict))


def add_env_command(cli: click.Group) -> None:
    from bentoml import __version__ as BENTOML_VERSION
    from bentoml.exceptions import CLIException
    from bentoml._internal.utils.pkg import get_pkg_version
    from bentoml._internal.utils.pkg import PackageNotFoundError

    @cli.command(help=gettext("Print environment info and exit"))
    @click.option(
        "-o",
        "--output",
        type=click.Choice(["md", "bash"]),
        default="md",
        show_default=True,
        help="Output format. '-o bash' to display without format.",
    )
    @click.pass_context
    def env(ctx: click.Context, output: t.Literal["md", "bash"]) -> None:  # type: ignore (unused warning)
        if output not in ["md", "bash"]:
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
