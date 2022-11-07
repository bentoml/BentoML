from __future__ import annotations

import os
import shutil
import typing as t
import logging
from typing import TYPE_CHECKING

from .base import Arguments
from ..utils import resolve_user_filepath

if TYPE_CHECKING:
    from .base import ArgType
    from ..types import PathType

logger = logging.getLogger(__name__)

__all__ = [
    "health",
    "construct_build_args",
    "BUILDKIT_SUPPORT",
    "BUILD_CMD",
    "find_binary",
]

BUILDKIT_SUPPORT = False

BUILD_CMD = []


def find_binary() -> str | None:
    # We will check if /kaniko/executor exists
    if os.path.exists("/kaniko/executor"):
        return "/kaniko/executor"
    return shutil.which("executor")


def health() -> bool:
    client = find_binary()
    if client is None:
        logger.warning(
            "Kaniko executor not found. See https://github.com/GoogleContainerTools/kaniko."
        )
        return False
    return True


def parse_dict_opt(d: dict[str, str]) -> str:
    return ",".join([f"{key}={value}" for key, value in d.items()])


def construct_build_args(
    *,
    file: PathType | None = None,
    context_path: PathType = ".",
    build_arg: dict[str, str] | ArgType = None,
    destination: ArgType = None,
    no_push: bool = False,
    tar_path: PathType | None = None,
    label: dict[str, str] | ArgType = None,
    git: str | dict[str, str] | None = None,
    registry_certificate: dict[str, str] | ArgType = None,
    **kwargs: t.Any,
) -> Arguments:
    cmds = Arguments()
    tag = kwargs.pop("tag", ())[0]
    # NOTE: if you are running this in k8s cluster, set in_k8s to True.
    in_k8s = kwargs.pop("in_k8s", False)

    if not in_k8s:
        # Since we are building this from local dir and not k8s, our context
        # would look like dir://<context_path>
        context_path = f"dir://{resolve_user_filepath(str(context_path), ctx=None)}"
    cmds.construct_args(context_path, opt="context")
    cmds.construct_args(str(file), opt="dockerfile")
    if destination is None:
        destination = tag
        if tar_path is None:
            tar_path = os.path.join(os.getcwd(), f"{tag.replace(':', '_')}.tar")
        no_push = True
    cmds.construct_args(destination, opt="destination")
    cmds.construct_args(tar_path, opt="tar-path")
    cmds.construct_args(no_push, opt="no-push")

    if isinstance(build_arg, dict):
        build_arg = tuple(f"{key}={value}" for key, value in build_arg.items())
    cmds.construct_args(build_arg, opt="build-arg")
    if isinstance(registry_certificate, dict):
        registry_certificate = tuple(
            f"{key}={value}" for key, value in registry_certificate.items()
        )
    cmds.construct_args(registry_certificate, opt="registry-certificate")
    if isinstance(label, dict):
        label = tuple(f"{key}={value}" for key, value in label.items())
    cmds.construct_args(label, opt="label")
    if isinstance(git, dict):
        git = parse_dict_opt(git)
    cmds.construct_args(git, opt="git")

    for k, v in kwargs.items():
        cmds.construct_args(v, opt=k.replace("_", "-"))

    return cmds
