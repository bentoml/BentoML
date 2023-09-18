from __future__ import annotations

import os
import re
import shutil
import typing as t
import logging
import subprocess
from typing import TYPE_CHECKING

import psutil
from packaging.version import parse

from .base import Arguments

if TYPE_CHECKING:
    from .base import ArgType
    from ..types import PathType

logger = logging.getLogger(__name__)

__all__ = ["ENV", "health", "construct_build_args", "BUILDKIT_SUPPORT", "find_binary"]


ENV = {"DOCKER_BUILDKIT": "1", "DOCKER_SCAN_SUGGEST": "false"}

BUILDKIT_SUPPORT = True


def find_binary() -> str | None:
    return shutil.which("docker")


def health() -> bool:
    client = find_binary()
    if client is None:
        logger.warning(
            "Docker not found. Make sure it is installed and in your PATH. See https://docs.docker.com/get-docker/"
        )
        return False
    # next, we will check if given docker is > 18.09 for BuildKit support.
    v = re.match(
        r"(\d+)\.(\d+)\.(\d+)",  # handle pre-release versions of docker.
        subprocess.check_output(
            [client, "version", "--format", "{{json .Server.Version}}"]
        )
        .decode("utf-8")
        .strip('\n"'),
    )
    assert v is not None
    if parse(v.group()) < parse("18.09") and int(os.getenv("DOCKER_BUILDKIT", 1)) == 1:
        logger.warning(
            "Docker version: %s, which doesn't support BuildKit. Please upgrade to 18.09 or newer. See https://docs.docker.com/get-docker/",
            v.group(),
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
    cache_from: str | dict[str, str] | ArgType = None,
    isolation: t.Literal["default", "process", "hyperv"] | None = None,
    label: dict[str, str] | ArgType = None,
    output: str | dict[str, str] | ArgType = None,
    secret: str | dict[str, str] | ArgType = None,
    **kwargs: t.Any,
) -> Arguments:
    cmds = Arguments()

    if isinstance(add_host, dict):
        add_host = tuple(f"{host}:{ip}" for host, ip in add_host.items())
    cmds.construct_args(add_host, opt="add-host")
    if isinstance(build_arg, dict):
        build_arg = tuple(f"{key}={value}" for key, value in build_arg.items())
    cmds.construct_args(build_arg, opt="build-arg")
    if isinstance(cache_from, dict):
        cache_from = parse_dict_opt(cache_from)
    cmds.construct_args(cache_from, opt="cache-from")
    if isolation is not None:
        if not psutil.WINDOWS and isolation != "default":
            logger.warning("'isolation=%s' is only supported on Windows.", isolation)
        cmds.construct_args(isolation, opt="isolation")
    if isinstance(label, dict):
        label = tuple(f"{key}={value}" for key, value in label.items())
    cmds.construct_args(label, opt="label")
    if isinstance(output, dict):
        output = parse_dict_opt(output)
    cmds.construct_args(output, opt="output")
    if isinstance(secret, dict):
        secret = parse_dict_opt(secret)
    cmds.construct_args(secret, opt="secret")

    for k, v in kwargs.items():
        cmds.construct_args(v, opt=k.replace("_", "-"))

    cmds.append(str(context_path))

    return cmds
