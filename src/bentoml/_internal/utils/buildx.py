from __future__ import annotations

import os
import typing as t
import logging
import subprocess
from typing import TYPE_CHECKING

from ..types import PathType
from ...exceptions import BentoMLException

if TYPE_CHECKING:
    P = t.ParamSpec("P")

logger = logging.getLogger(__name__)


def health() -> None:
    """
    Check whether buildx is available in given system.
    """
    cmds = ["docker", "build", "--help"]
    try:
        output = subprocess.check_output(cmds)
        assert "Build an image" in output.decode("utf-8")
    except (subprocess.CalledProcessError, AssertionError) as e:
        raise BentoMLException(f"Failed to run '{cmds}': {e}")


def build(
    subprocess_env: dict[str, str] | None,
    cwd: PathType | None,
    *,
    context_path: PathType = ".",
    add_host: dict[str, str] | None,
    build_args: dict[str, str] | None,
    cache_from: str | tuple[str] | dict[str, str] | None,
    file: PathType | None,
    iidfile: PathType | None,
    labels: dict[str, str] | None,
    network: str | None,
    no_cache: bool,
    output: str | dict[str, str] | None,
    platform: str | tuple[str] | None,
    progress: t.Literal["auto", "tty", "plain"],
    pull: bool,
    quiet: bool,
    secrets: str | t.Iterable[str] | None,
    ssh: str | None,
    tags: str | t.Iterable[str],
    target: str | None,
) -> None:
    cmds = ["docker", "build"]

    cmds += ["--progress", progress]

    if isinstance(tags, str):
        cmds.extend(["--tag", tags])
    elif isinstance(tags, tuple):
        for tag in tags:
            cmds.extend(["--tag", tag])

    if add_host is not None:
        hosts = [f"{k}:{v}" for k, v in add_host.items()]
        for host in hosts:
            cmds.extend(["--add-host", host])

    if platform is not None and len(platform) > 0:
        if isinstance(platform, str):
            platform = (platform,)
        cmds += ["--platform", ",".join(platform)]

    if build_args is not None:
        args = [f"{k}={v}" for k, v in build_args.items()]
        for arg in args:
            cmds.extend(["--build-arg", arg])

    if cache_from is not None:
        if isinstance(cache_from, str):
            cmds.extend(["--cache-from", cache_from])
        elif isinstance(cache_from, tuple):
            for arg in cache_from:
                cmds.extend(["--cache-from", arg])
        else:
            args = [f"{k}={v}" for k, v in cache_from.items()]
            cmds.extend(["--cache-from", ",".join(args)])

    if file is not None:
        cmds.extend(["--file", str(file)])

    if iidfile is not None:
        cmds.extend(["--iidfile", str(iidfile)])

    if network is not None:
        cmds.extend(["--network", network])

    if no_cache:
        cmds.append("--no-cache")

    if labels is not None:
        args = [f"{k}={v}" for k, v in labels.items()]
        for arg in args:
            cmds.extend(["--label", arg])

    if output is not None:
        if isinstance(output, str):
            cmds.extend(["--output", output])
        else:
            args = [f"{k}={v}" for k, v in output.items()]
            cmds += ["--output", ",".join(args)]

    if pull:
        cmds.append("--pull")

    if quiet:
        cmds.append("--quiet")

    if secrets is not None:
        if isinstance(secrets, str):
            cmds.extend(["--secret", secrets])
        else:
            for secret in secrets:
                cmds.extend(["--secret", secret])

    if ssh is not None:
        cmds.extend(["--ssh", ssh])

    if target is not None:
        cmds.extend(["--target", target])

    cmds.append(str(context_path))

    logger.debug("docker build cmd: '%s'", cmds)

    run_docker_cmd(cmds, env=subprocess_env, cwd=cwd)


def run_docker_cmd(
    cmds: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: PathType | None = None,
) -> None:
    subprocess_env = os.environ.copy()
    if env is not None:
        subprocess_env.update(env)

    subprocess.check_output(list(map(str, cmds)), env=subprocess_env, cwd=cwd)
