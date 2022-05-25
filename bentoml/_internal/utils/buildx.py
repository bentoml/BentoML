from __future__ import annotations

import os
import typing as t
import logging
import functools
import subprocess
from typing import TYPE_CHECKING

from ..types import PathType
from ...exceptions import BentoMLException

if TYPE_CHECKING:
    P = t.ParamSpec("P")

logger = logging.getLogger(__name__)

DOCKER_BUILDX_CMD = ["docker", "buildx"]

# https://stackoverflow.com/questions/45125516/possible-values-for-uname-m
UNAME_M_TO_PLATFORM_MAPPING = {
    "x86_64": "linux/amd64",
    "arm64": "linux/arm64/v8",
    "ppc64le": "linux/ppc64le",
    "s390x": "linux/s390x",
    "riscv64": "linux/riscv64",
    "mips64": "linux/mips64le",
}


@functools.lru_cache(maxsize=1)
def health() -> None:
    """
    Check whether buildx is available in given system.
    """
    cmds = DOCKER_BUILDX_CMD + ["--help"]
    try:
        output = subprocess.check_output(cmds, stderr=subprocess.STDOUT)
        assert "buildx" in output.decode("utf-8")
    except (subprocess.CalledProcessError, AssertionError):
        raise BentoMLException(
            "BentoML requires Docker Buildx to be installed to support multi-arch builds. "
            "Buildx comes with Docker Desktop, but one can also install it manually by following "
            "instructions via https://docs.docker.com/buildx/working-with-buildx/#install."
        )


def build(
    subprocess_env: dict[str, str] | None,
    cwd: PathType | None,
    *,
    context_path: PathType = ".",
    add_host: dict[str, str] | None,
    allow: list[str] | None,
    build_args: dict[str, str] | None,
    build_context: dict[str, str] | None,
    builder: str | None,
    cache_from: str | list[str] | dict[str, str] | None,
    cache_to: str | list[str] | dict[str, str] | None,
    cgroup_parent: str | None,
    file: PathType | None,
    iidfile: PathType | None,
    labels: dict[str, str] | None,
    load: bool,
    metadata_file: PathType | None,
    network: str | None,
    no_cache: bool,
    no_cache_filter: list[str] | None,
    output: str | dict[str, str] | None,
    platform: str | list[str] | None,
    progress: t.Literal["auto", "tty", "plain"],
    pull: bool,
    push: bool,
    quiet: bool,
    secrets: str | list[str] | None,
    shm_size: str | int | None,
    rm: bool,
    ssh: str | None,
    tags: str | list[str] | None,
    target: str | None,
    ulimit: str | None,
) -> None:
    cmds = DOCKER_BUILDX_CMD + ["build"]

    cmds += ["--progress", progress]

    if tags is None:
        tags = []
    tags = [tags] if not isinstance(tags, list) else tags
    for tag in tags:
        cmds.extend(["--tag", tag])

    if add_host is not None:
        hosts = [f"{k}:{v}" for k, v in add_host.items()]
        for host in hosts:
            cmds.extend(["--add-host", host])

    if allow is not None:
        for allow_arg in allow:
            cmds.extend(["--allow", allow_arg])

    if build_args is not None:
        args = [f"{k}={v}" for k, v in build_args.items()]
        for arg in args:
            cmds.extend(["--build-arg", arg])

    if build_context is not None:
        args = [f"{k}={v}" for k, v in build_context.items()]
        for arg in args:
            cmds.extend(["--build-context", arg])

    if builder is not None:
        cmds.extend(["--builder", builder])

    if cache_from is not None:
        if isinstance(cache_from, str):
            cmds.extend(["--cache-from", cache_from])
        elif isinstance(cache_from, list):
            for arg in cache_from:
                cmds.extend(["--cache-from", arg])
        else:
            args = [f"{k}={v}" for k, v in cache_from.items()]
            cmds.extend(["--cache-from", ",".join(args)])

    if cache_to is not None:
        if isinstance(cache_to, str):
            cmds.extend(["--cache-to", cache_to])
        elif isinstance(cache_to, list):
            for arg in cache_to:
                cmds.extend(["--cache-to", arg])
        else:
            args = [f"{k}={v}" for k, v in cache_to.items()]
            cmds.extend(["--cache-to", ",".join(args)])

    if cgroup_parent is not None:
        cmds.extend(["--cgroup-parent", cgroup_parent])

    if file is not None:
        cmds.extend(["--file", str(file)])

    if iidfile is not None:
        cmds.extend(["--iidfile", str(iidfile)])

    if load:
        cmds.append("--load")

    if metadata_file is not None:
        cmds.extend(["--metadata-file", str(metadata_file)])

    if network is not None:
        cmds.extend(["--network", network])

    if no_cache:
        cmds.append("--no-cache")

    if no_cache_filter is not None:
        for arg in no_cache_filter:
            cmds.extend(["--no-cache-filter", arg])

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

    if platform:
        if isinstance(platform, str):
            platform = [platform]
        cmds += ["--platform", ",".join(platform)]

    if pull:
        cmds.append("--pull")

    if push:
        cmds.append("--push")

    if quiet:
        cmds.append("--quiet")

    if secrets is not None:
        if isinstance(secrets, str):
            cmds.extend(["--secret", secrets])
        else:
            for secret in secrets:
                cmds.extend(["--secret", secret])

    if rm:
        cmds.append("--rm")

    if shm_size is not None:
        cmds.extend(["--shm-size", str(shm_size)])

    if ssh is not None:
        cmds.extend(["--ssh", ssh])

    if target is not None:
        cmds.extend(["--target", target])

    if ulimit is not None:
        cmds.extend(["--ulimit", ulimit])

    cmds.append(str(context_path))

    logger.debug(f"docker buildx build cmd: [bold yellow]{cmds}[/]")

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
