from __future__ import annotations

import os
import shutil
import typing as t
import logging
from typing import TYPE_CHECKING

import psutil

from .base import Arguments

if TYPE_CHECKING:
    from .base import ArgType
    from ..types import PathType

logger = logging.getLogger(__name__)

__all__ = ["health", "construct_build_args", "BUILDKIT_SUPPORT", "find_binary"]


BUILDKIT_SUPPORT = True


def find_binary() -> str | None:
    return shutil.which("nerdctl")


def health() -> bool:
    client = find_binary()

    if psutil.WINDOWS:
        logger.error("nerdctl is NOT SUPPORTED on Windows.")
        return False

    if os.getenv("DOCKER_BUILDKIT", "") != "":
        logger.warning("DOCKER_BUILDKIT will have no effect when using nerdctl.")

    if client is None:
        logger.warning(
            "nerdctl not found. Make sure it is installed and in your PATH. See https://github.com/containerd/nerdctl"
        )
        if psutil.MACOS:
            logger.info(
                "To run nerdctl on MacOS, use lima. See https://github.com/lima-vm/lima#getting-started and use BuildKit templates to run nerdctl."
            )
        else:
            logger.info(
                "See https://github.com/containerd/nerdctl/blob/master/docs/build.md for setting BuildKit with nerdctl."
            )
        return False

    from .buildctl import buildkitd_health

    return buildkitd_health()


def parse_dict_opt(d: dict[str, str]) -> str:
    return ",".join([f"{key}={value}" for key, value in d.items()])


def construct_build_args(
    *,
    context_path: PathType = ".",
    build_arg: dict[str, str] | ArgType = None,
    cache_from: str | dict[str, str] | ArgType = None,
    cache_to: str | dict[str, str] | ArgType = None,
    label: dict[str, str] | ArgType = None,
    output: str | dict[str, str] | ArgType = None,
    secret: str | dict[str, str] | ArgType = None,
    **kwargs: t.Any,
) -> Arguments:
    cmds = Arguments()

    if isinstance(build_arg, dict):
        build_arg = tuple(f"{key}={value}" for key, value in build_arg.items())
    cmds.construct_args(build_arg, opt="build-arg")
    if isinstance(cache_from, dict):
        cache_from = parse_dict_opt(cache_from)
    cmds.construct_args(cache_from, opt="cache-from")
    if isinstance(cache_to, dict):
        cache_to = parse_dict_opt(cache_to)
    cmds.construct_args(cache_to, opt="cache-to")
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
