from __future__ import annotations

import logging
import shutil
import typing as t
from typing import TYPE_CHECKING

from .base import Arguments

if TYPE_CHECKING:
    from ..types import PathType
    from .base import ArgType

logger = logging.getLogger(__name__)

__all__ = ["ENV", "health", "construct_build_args", "BUILDKIT_SUPPORT", "find_binary"]

BUILDKIT_SUPPORT = False

# by default, the format will be oci (v2)
# user can override this by setting the format to docker
# via the environment variable BUILDAH_FORMAT
ENV = {"BUILDAH_FORMAT": "oci", "DOCKER_BUILDKIT": "0"}


def find_binary() -> str | None:
    return shutil.which("buildah")


def health() -> bool:
    client = find_binary()
    if client is None:
        logger.warning(
            "Buildah not found. Make sure it is installed and in your PATH. See https://github.com/containers/buildah/blob/main/install.md"
        )
        return False
    return True


def parse_dict_opt(d: dict[str, str]) -> str:
    return ",".join([f"{key}={value}" for key, value in d.items()])


def construct_build_args(
    *,
    context_path: PathType = ".",
    add_host: dict[str, str] | ArgType = None,
    annotation: dict[str, str] | ArgType = None,
    label: dict[str, str] | ArgType = None,
    build_arg: dict[str, str] | ArgType = None,
    creds: str | dict[str, str] | ArgType = None,
    decryption_key: str | dict[str, str] | ArgType = None,
    runtime_flag: str | dict[str, str] | ArgType = None,
    secret: str | dict[str, str] | ArgType = None,
    ulimit: str | dict[str, tuple[int, int]] | ArgType = None,
    volume: str | tuple[str, str, str] | None = None,
    **kwargs: t.Any,
) -> Arguments:
    cmds = Arguments()

    if isinstance(add_host, dict):
        add_host = tuple(f"{host}:{ip}" for host, ip in add_host.items())
    cmds.construct_args(add_host, opt="add-host")
    if isinstance(label, dict):
        label = tuple(f"{key}={value}" for key, value in label.items())
    cmds.construct_args(label, opt="label")
    if isinstance(annotation, dict):
        annotation = tuple(f"{key}={value}" for key, value in annotation.items())
    cmds.construct_args(annotation, opt="annotation")
    if isinstance(build_arg, dict):
        build_arg = tuple(f"{key}={value}" for key, value in build_arg.items())
    cmds.construct_args(build_arg, opt="build-arg")
    if isinstance(creds, dict):
        creds = tuple(f"{key}:{value}" for key, value in creds.items())
    cmds.construct_args(creds, opt="creds")
    if isinstance(decryption_key, dict):
        decryption_key = tuple(
            f"{key}:{value}" for key, value in decryption_key.items()
        )
    cmds.construct_args(decryption_key, opt="decryption-key")
    if isinstance(runtime_flag, dict):
        runtime_flag = tuple(f"{key}={value}" for key, value in runtime_flag.items())
    cmds.construct_args(runtime_flag, opt="runtime-flag")
    if isinstance(secret, dict):
        secret = parse_dict_opt(secret)
    cmds.construct_args(secret, opt="secret")
    if isinstance(ulimit, dict):
        ulimit = tuple(f"{key}={value[0]}:{value[1]}" for key, value in ulimit.items())
    cmds.construct_args(ulimit, opt="ulimit")
    if isinstance(volume, tuple):
        volume = ":".join(volume)
    cmds.construct_args(volume, opt="volume")

    for k, v in kwargs.items():
        cmds.construct_args(v, opt=k.replace("_", "-"))

    cmds.append(str(context_path))

    return cmds
