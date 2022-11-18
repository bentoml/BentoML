from __future__ import annotations

import typing as t
import logging
import subprocess
from typing import TYPE_CHECKING

from .base import Arguments
from .docker import ENV
from .docker import health as _docker_health
from .docker import find_binary

if TYPE_CHECKING:
    from .base import ArgType
    from ..types import PathType

logger = logging.getLogger(__name__)

__all__ = ["ENV", "health", "construct_build_args", "BUILDKIT_SUPPORT", "BUILD_CMD"]

BUILDKIT_SUPPORT = True

BUILD_CMD = ["buildx", "build"]


def health() -> bool:
    if not _docker_health():
        return False
    client = find_binary()
    assert client is not None
    has_buildx = subprocess.check_output([client, "buildx", "--help"]).decode("utf-8")
    if "--builder string" not in has_buildx:
        logger.warning(
            "Buildx is not installed. See https://docs.docker.com/build/buildx/install/ for instalation instruction."
        )
        return False
    return True


def parse_dict_opt(d: dict[str, str]) -> str:
    return ",".join([f"{key}={value}" for key, value in d.items()])


def construct_build_args(
    *,
    context_path: PathType = ".",
    add_host: dict[str, str] | ArgType = None,
    build_arg: dict[str, str] | ArgType = None,
    build_context: dict[str, str] | ArgType = None,
    cache_from: str | dict[str, str] | ArgType = None,
    cache_to: str | dict[str, str] | ArgType = None,
    label: dict[str, str] | ArgType = None,
    load: bool = True,
    no_cache_filter: str | dict[str, str] | ArgType = None,
    output: str | dict[str, str] | ArgType = None,
    platform: str | ArgType = None,
    push: bool = False,
    secret: str | dict[str, str] | ArgType = None,
    ulimit: str | dict[str, tuple[int, int]] | ArgType = None,
    **kwargs: t.Any,
) -> Arguments:
    cmds = Arguments()

    if platform and len(platform) > 1:
        if not push:
            logger.warning(
                "Multiple '--platform' arguments were found. Make sure to also use '--push' to push images to a repository or generated images will not be saved. See https://docs.docker.com/engine/reference/commandline/buildx_build/#load."
            )
    if push:
        load = False
    if isinstance(output, dict):
        output = parse_dict_opt(output)
    if output is not None and any("local" in o for o in output):
        load, push = False, False
    cmds.construct_args(output, opt="output")
    cmds.construct_args(push, opt="push")
    cmds.construct_args(load, opt="load")
    cmds.construct_args(platform, opt="platform")

    if isinstance(add_host, dict):
        add_host = tuple(f"{host}:{ip}" for host, ip in add_host.items())
    cmds.construct_args(add_host, opt="add-host")
    if isinstance(build_arg, dict):
        build_arg = tuple(f"{key}={value}" for key, value in build_arg.items())
    cmds.construct_args(build_arg, opt="build-arg")
    if isinstance(build_context, dict):
        build_context = tuple(f"{key}={value}" for key, value in build_context.items())
    cmds.construct_args(build_context, opt="build-context")
    if isinstance(cache_from, dict):
        cache_from = parse_dict_opt(cache_from)
    cmds.construct_args(cache_from, opt="cache-from")
    if isinstance(cache_to, dict):
        cache_to = parse_dict_opt(cache_to)
    cmds.construct_args(cache_to, opt="cache-to")
    if isinstance(label, dict):
        label = tuple(f"{key}={value}" for key, value in label.items())
    cmds.construct_args(label, opt="label")
    if isinstance(no_cache_filter, dict):
        no_cache_filter = parse_dict_opt(no_cache_filter)
    cmds.construct_args(no_cache_filter, opt="no-cache-filter")
    if isinstance(secret, dict):
        secret = parse_dict_opt(secret)
    cmds.construct_args(secret, opt="secret")
    if isinstance(ulimit, dict):
        # {"cpu": (1023, 1024), "fsize": (8192, 8192)}
        ulimit = tuple(f"{key}={value[0]}:{value[1]}" for key, value in ulimit.items())
    cmds.construct_args(ulimit, opt="ulimit")

    for k, v in kwargs.items():
        cmds.construct_args(v, opt=k.replace("_", "-"))

    cmds.append(str(context_path))

    return cmds
