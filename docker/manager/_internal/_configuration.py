from __future__ import annotations

import os
import re
import typing as t
import logging
from typing import TYPE_CHECKING
from pathlib import Path

import fs
import yaml
from simple_di import inject
from simple_di import Provide
from simple_di import container
from simple_di.providers import SingletonFactory

from .exceptions import ManagerException

if TYPE_CHECKING:
    from fs.base import FS

    GenericDict = t.Dict[str, t.Any]

logger = logging.getLogger(__name__)

#########################################################

SUPPORTED_PYTHON_VERSION = ["3.6", "3.7", "3.8", "3.9", "3.10"]
DOCKERFILE_BUILD_HIERARCHY = ("base", "runtime", "cudnn", "devel")
SUPPORTED_ARCHITECTURE_TYPE = ["amd64", "arm64v8", "ppc64le", "s390x"]
SUPPORTED_OS_RELEASES = [
    "ubi8",
    "amazonlinux2",
    "alpine3.14",
    "debian11",
    "debian10",
]

DOCKER_DIRECTORY = Path(os.path.dirname(__file__)).parent.parent
# release entry should always include CUDA support :)
RELEASE_KEY_RGX = re.compile(r"w_cuda_v([\d\.]+)(?:_(\w+))?$")
# We only care about linux mapping to uname
DOCKER_TARGETARCH_LINUX_UNAME_ARCH_MAPPING = {
    "amd64": "x86_64",
    "arm64v8": "aarch64",
    "arm32v5": "armv7l",
    "arm32v6": "armv7l",
    "arm32v7": "armv7l",
    "i386": "386",
    "ppc64le": "ppc64le",
    "s390x": "s390x",
    "riscv64": "riscv64",
    "mips64le": "mips64",
}


@container
class ManagerContainerClass:

    default_name = "bento-server"

    @SingletonFactory
    @staticmethod
    def root_fs() -> FS:
        return fs.open_fs(DOCKER_DIRECTORY.__fspath__())

    @SingletonFactory
    @staticmethod
    def default_context() -> GenericDict:
        return get_manifest_info("bento-server", "11.5.1")


DockerManagerContainer = ManagerContainerClass()


@inject
def get_manifest_info(
    docker_package: str,
    cuda_version: str,
    *,
    docker_fs_: "FS" = Provide[DockerManagerContainer.root_fs],
) -> GenericDict:
    manifest_fs = docker_fs_.opendir("manifest")
    fname = f"{docker_package}.cuda_v{cuda_version}.yaml"
    if not manifest_fs.exists(fname):
        raise ManagerException(f"{fname} doesn't exist under manifest directory.")

    with manifest_fs.open(fname, "r", encoding="utf-8") as f:
        manifest = yaml.load(f.read(), Loader=yaml.FullLoader)
        manifest.pop("common")

    distros = {}
    for key in manifest.copy():
        if RELEASE_KEY_RGX.match(key):
            distros[key] = manifest.pop(key)
    return distros
