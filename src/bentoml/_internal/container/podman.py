from __future__ import annotations

import shutil
import typing as t
import logging
import subprocess
from typing import TYPE_CHECKING

import psutil

from .base import Arguments
from .buildah import ENV

if TYPE_CHECKING:
    from .base import ArgType
    from ..types import PathType

logger = logging.getLogger(__name__)

__all__ = ["ENV", "health", "construct_build_args", "BUILDKIT_SUPPORT", "find_binary"]


BUILDKIT_SUPPORT = False


def find_binary() -> str | None:
    return shutil.which("podman")


def health() -> bool:
    client = find_binary()
    if client is None:
        logger.warning(
            "Podman not found. Make sure it is installed and in your PATH. See https://podman.io/getting-started/installation"
        )
        return False

    if psutil.MACOS or psutil.WINDOWS:
        # check if podman machine is running.
        output = (
            subprocess.check_output(
                [client, "machine", "info", "--format", "{{json .Host.MachineState}}"]
            )
            .decode("utf-8")
            .strip("\n&#34;")  # strip quotation marks from JSON format.
            .lower()
        )
        return output == '"running"'
    else:
        return True


def parse_dict_opt(d: dict[str, str]) -> str:
    return ",".join([f"{key}={value}" for key, value in d.items()])


def construct_build_args(
    *,
    context_path: PathType = ".",
    add_host: dict[str, str] | ArgType = None,
    all_platforms: bool = False,
    annotation: dict[str, str] | ArgType = None,
    label: dict[str, str] | ArgType = None,
    build_arg: dict[str, str] | ArgType = None,
    build_context: dict[str, str] | ArgType = None,
    creds: str | dict[str, str] | ArgType = None,
    decryption_key: str | dict[str, str] | ArgType = None,
    env: str | dict[str, str] | ArgType = None,
    output: str | dict[str, str] | ArgType = None,
    runtime_flag: str | dict[str, str] | ArgType = None,
    secret: str | dict[str, str] | ArgType = None,
    ulimit: str | dict[str, tuple[int, int]] | ArgType = None,
    userns_gid_map: str | tuple[str, str, str] | None = None,
    userns_uid_map: str | tuple[str, str, str] | None = None,
    volume: str | tuple[str, str, str] | None = None,
    **kwargs: t.Any,
) -> Arguments:
    cmds = Arguments()

    if isinstance(add_host, dict):
        add_host = tuple(f"{host}:{ip}" for host, ip in add_host.items())
    cmds.construct_args(add_host, opt="add-host")
    cmds.construct_args(all_platforms, opt="all-platforms")
    if isinstance(label, dict):
        label = tuple(f"{key}={value}" for key, value in label.items())
    cmds.construct_args(label, opt="label")
    if isinstance(annotation, dict):
        annotation = tuple(f"{key}={value}" for key, value in annotation.items())
    cmds.construct_args(annotation, opt="annotation")
    if isinstance(build_arg, dict):
        build_arg = tuple(f"{key}={value}" for key, value in build_arg.items())
    cmds.construct_args(build_arg, opt="build-arg")
    if isinstance(build_context, dict):
        build_context = tuple(f"{key}={value}" for key, value in build_context.items())
    cmds.construct_args(build_context, opt="build-context")
    if isinstance(creds, dict):
        creds = tuple(f"{key}:{value}" for key, value in creds.items())
    cmds.construct_args(creds, opt="creds")
    if isinstance(decryption_key, dict):
        decryption_key = tuple(
            f"{key}:{value}" for key, value in decryption_key.items()
        )
    cmds.construct_args(decryption_key, opt="decryption-key")
    if isinstance(env, dict):
        env = tuple(f"{key}={value}" for key, value in env.items())
    cmds.construct_args(env, opt="env")
    if isinstance(output, dict):
        output = parse_dict_opt(output)
    cmds.construct_args(output, opt="output")
    if isinstance(runtime_flag, dict):
        runtime_flag = tuple(f"{key}={value}" for key, value in runtime_flag.items())
    cmds.construct_args(runtime_flag, opt="runtime-flag")
    if isinstance(secret, dict):
        secret = parse_dict_opt(secret)
    cmds.construct_args(secret, opt="secret")
    if isinstance(ulimit, dict):
        ulimit = tuple(f"{key}={value[0]}:{value[1]}" for key, value in ulimit.items())
    cmds.construct_args(ulimit, opt="ulimit")
    if isinstance(userns_gid_map, tuple):
        userns_gid_map = ":".join(userns_gid_map)
    cmds.construct_args(userns_gid_map, opt="userns-gid-map")
    if isinstance(userns_uid_map, tuple):
        userns_uid_map = ":".join(userns_uid_map)
    cmds.construct_args(userns_uid_map, opt="userns-uid-map")
    if isinstance(volume, tuple):
        volume = ":".join(volume)
    cmds.construct_args(volume, opt="volume")
    for k, v in kwargs.items():
        cmds.construct_args(v, opt=k.replace("_", "-"))

    cmds.append(str(context_path))

    return cmds
