from __future__ import annotations

import os
import re
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
    return shutil.which("buildctl")


def buildkitd_health() -> bool:
    if shutil.which("buildkitd") or os.getenv("BUILDKIT_HOST", "") != "":
        pass
    else:
        logger.error(
            "Both buildkitd and BUILDKIT_HOST are not found. Ensure to use either of them."
        )
        tool = shutil.which("docker") or shutil.which("podman")
        if tool is not None:
            buildkitd_instructions = "\n".join(
                [
                    "To run a portable buildkitd (via container), do the following in a terminal:",
                    "    %s run -d --name buildkitd --privileged moby/buildkit:master",
                    "    export BUILDKIT_HOST=%s-container://buildkitd",
                    "Then proceed with 'containerize' again.",
                ]
            )
            logger.info(buildkitd_instructions, tool, tool.split(os.sep)[-1])
        return False

    return True


def health() -> bool:
    client = find_binary()

    if psutil.WINDOWS:
        logger.error("buildctl is NOT SUPPORTED on Windows.")
        return False

    if not buildkitd_health():
        return False

    if client is None:
        logger.warning(
            "buildctl not found. Make sure it is installed and in your PATH. See https://github.com/moby/buildkit"
        )
        return False

    return True


def parse_dict_opt(d: dict[str, str]) -> str:
    return ",".join([f"{key}={value}" for key, value in d.items()])


ctl_args = (
    "output",
    "progress",
    "trace",
    "local",
    "frontend",
    "opt",
    "no-cache",
    "export-cache",
    "import-cache",
    "secret",
    "allow",
    "ssh",
    "metadata-file",
)

is_ctl_arg = re.compile(r"^(" + "|".join(ctl_args) + ")").search


def construct_build_args(
    *,
    file: PathType | None = None,
    tag: tuple[str] | None = None,
    context_path: PathType = ".",
    output: str | dict[str, str] | ArgType = None,
    **kwargs: t.Any,
) -> Arguments:
    # NOTE: In the case of buildctl, we will have to parse the options manually from kwargs.
    cmds = Arguments()

    opt_args: dict[str, t.Any] = {}
    buildctl_args = {
        k: v for k, v in kwargs.items() if v and is_ctl_arg(k.replace("_", "-"))
    }

    local: dict[str, str] = buildctl_args.pop("local", {})
    if isinstance(local, tuple):
        local = {k: v for (k, v) in map(lambda s: tuple(s.split("=")), local)}
    if "context" in local:
        logger.warning(
            "Passing 'context' to 'local' is not supported. If you are using Python API, use 'context_path' instead."
        )
    local["context"] = str(context_path)
    if "dockerfile" in local and file is not None:
        logger.warning(
            "Passing 'dockerfile' to 'local' is not supported. If you are using Python API, use 'file' instead."
        )
    local["dockerfile"] = os.path.dirname(str(file))
    cmds.construct_args(tuple(f"{k}={v}" for k, v in local.items()), opt="local")

    frontend: str = buildctl_args.pop("frontend", "dockerfile.v0")
    if isinstance(frontend, tuple):
        if len(frontend) > 1:
            logger.warning(
                "Multiple frontend options are not supported. Using the first one."
            )
        frontend = frontend[0]
    cmds.construct_args(frontend, opt="frontend")
    if frontend == "gateway.v0":
        opt_args["source"] = "docker/dockerfile"

    if isinstance(output, dict):
        output = parse_dict_opt(output)
    cmds.construct_args(output, opt="output")

    # handle buildctl arguments
    for k, v in buildctl_args.items():
        cmds.construct_args(v, opt=k.replace("_", "-"))

    # handle --opt
    opt_args.update(
        {k: v for k, v in kwargs.items() if is_ctl_arg(k.replace("_", "-")) is None}
    )
    for k, v in opt_args.items():
        if isinstance(v, tuple):
            cmds.construct_args(tuple(map(lambda it: f"{k}:{it}", v)), opt="opt")
        elif isinstance(v, str):
            cmds.construct_args(f"{k}={v}", opt="opt")
        else:
            raise ValueError(f"Unsupported type for {k}: {type(v)}")

    if tag is not None:
        if output is None:
            logger.warning(
                "Autoconfig for output type is deprecated and will be removed in the next major release. See message below."
            )
            # NOTE: We will always use the docker image spec if docker is available.
            # Otherwise fallback to the OCI image spec.
            if shutil.which("docker") is not None:
                cmds.construct_args(
                    tuple(map(lambda tg: f"type=docker,name=docker.io/{tg}", tag)),
                    opt="output",
                )
            else:
                cmds.construct_args(
                    tuple(map(lambda tg: f"type=oci,name={tg}", tag)),
                    opt="output",
                )
    else:
        logger.info(
            "'tag' is not specified. Result image will only be saved in build cache."
        )

    return cmds
