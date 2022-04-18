from __future__ import annotations

import os
import typing as t
import logging
import importlib.metadata
from typing import TYPE_CHECKING
from pathlib import Path

import fs
import yaml
from simple_di import inject
from simple_di import Provide
from simple_di import container
from simple_di.providers import SingletonFactory

if TYPE_CHECKING:
    from fs.base import FS

    GenericDict = t.Dict[str, t.Any]

logger = logging.getLogger(__name__)


DOCKER_DIRECTORY = Path(os.path.dirname(__file__)).parent
CUDA_VERSION = "11.5.1"


@container
class ManagerContainerClass:

    SUPPORTED_PYTHON_VERSION = ("3.7", "3.8", "3.9", "3.10")
    SUPPORTED_ARCHITECTURE_TYPE = ["amd64", "arm64v8", "ppc64le"]
    SUPPORTED_DISTRO_TYPE = [
        "ubi8",
        "amazonlinux2",
        "alpine3.14",
        "debian11",
        "debian10",
    ]
    RELEASE_TYPE_HIERARCHY = ("base", "runtime", "cudnn", "devel")

    docker_package = "bento-server"
    organization = "bentoml"

    @SingletonFactory
    @staticmethod
    def cuda_version() -> str:
        return os.environ.get("CUDA_VERSION", CUDA_VERSION)

    @SingletonFactory
    @staticmethod
    def bentoml_version() -> str:
        try:
            return os.environ["DOCKER_BENTOML_VERSION"]
        except KeyError:
            version = importlib.metadata.version("bentoml").split("+")[0]
            version = version.rsplit(".", maxsplit=1)[0]
            return version

    @SingletonFactory
    @staticmethod
    def root_fs() -> FS:
        return fs.open_fs(DOCKER_DIRECTORY.__fspath__())

    @SingletonFactory
    @staticmethod
    def templates_fs() -> FS:
        return fs.open_fs(DOCKER_DIRECTORY.joinpath("templates").__fspath__())

    @SingletonFactory
    @staticmethod
    def generated_fs() -> FS:
        return fs.open_fs(DOCKER_DIRECTORY.joinpath("generated").__fspath__())


DockerManagerContainer = ManagerContainerClass()


@inject
def get_manifest_info(
    *,
    cuda_version_: str = Provide[DockerManagerContainer.cuda_version],
    docker_fs_: FS = Provide[DockerManagerContainer.root_fs],
) -> GenericDict:
    manifest_fs = docker_fs_.opendir("manifest")
    fname = f"bento-server.cuda_v{cuda_version_}.yaml"
    if not manifest_fs.exists(fname):
        raise FileNotFoundError(f"{fname} doesn't exist under manifest directory.")

    with manifest_fs.open(fname, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f.read())
        manifest.pop("common")

    return manifest
