import logging
import os
import typing as t
from contextlib import contextmanager

import fs
import fs.copy
from fs.base import FS
from typing_extensions import Literal

from ..types import PathType
from .docker import ImageProvider
from .templates import BENTO_SERVER_DOCKERFILE

logger = logging.getLogger(__name__)


@contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


class DockerOptions:
    def __init__(
        self,
        # Options for choosing a BentoML built-in docker images
        distro: t.Optional[
            Literal["slim", "amazonlinux2", "alpine", "centos7", "centos8"]
        ] = None,
        python_version: t.Optional[str] = None,
        gpu: t.Optional[bool] = None,
        # A python or shell script that executes during docker build time
        setup_script: t.Optional[str] = None,
        # A user-provided custom docker image
        base_image: t.Optional[str] = None,
    ):
        if base_image is not None:
            self.base_image = base_image
            if distro is not None:
                logger.warning(
                    "docker base_image %s is used, 'distro=%s' option is ignored",
                    base_image,
                    distro,
                )
            if python_version is not None:
                logger.warning(
                    "docker base_image %s is used, 'python=%s' option is ignored",
                    base_image,
                    python_version,
                )
            if gpu is not None:
                logger.warning(
                    "docker base_image %s is used, 'gpu=%s' option is ignored",
                    base_image,
                    gpu,
                )
        else:
            self.base_image = repr(ImageProvider(distro, python_version, gpu))

            # TODO: remove this after 1.0 images published
            # Override to a fixed image for development purpose
            self.base_image = "bentoml/model-server:0.13.1-slim-py37"

        self.setup_script = setup_script

    def save(self, bento_fs: FS):
        docker_folder = fs.path.join("env", "docker")
        bento_fs.makedirs(docker_folder, recreate=True)
        dockerfile = fs.path.join(docker_folder, "Dockerfile")
        with bento_fs.open(dockerfile, "w") as dockerfile:
            dockerfile.write(BENTO_SERVER_DOCKERFILE.format(base_image=self.base_image))

        current_dir = fs.open_fs(os.path.dirname(__file__))
        for filename in ["bentoml-init.sh", "docker-entrypoint.sh"]:
            fs.copy.copy_file(
                current_dir, filename, bento_fs, fs.path.join(docker_folder, filename)
            )

        # TODO: copy over self.setup_script


class CondaOptions:
    def __init__(
        self,
        environment_yml_file: t.Optional[PathType] = None,
        channels: t.Optional[t.List[str]] = None,
        dependencies: t.Optional[t.List[str]] = None,
    ):
        ...

    def save(self, bento_fs: FS):
        ...


class PythonOptions:
    def __init__(
        self,
        version: t.Optional[str] = None,
        wheels: t.Optional[t.List[str]] = None,
        pip_install: t.Optional[t.Union[str, t.List[str]]] = None,
    ):
        ...

    def save(self, bento_fs: FS):
        ...


class BentoEnv:
    def __init__(
        self,
        # build ctx can be used to find files like requirements.txt or environment.yml
        build_ctx: PathType,
        # python PyPI dependencies
        python: t.Optional[t.Union[str, t.List[str]]] = None,
        # docker base image and build options
        docker: t.Optional[t.Dict[str, t.Any]] = None,
        # conda environment & dependencies
        conda: t.Optional[t.Union[str, t.Dict[str, t.Any]]] = None,
    ):
        self.build_ctx = build_ctx
        self.docker_options = (
            DockerOptions() if docker is None else DockerOptions(**docker)
        )
        self.python_options = (
            PythonOptions() if python is None else PythonOptions(**python)
        )
        if conda is None:
            self.conda_options = CondaOptions()
        elif isinstance(conda, str):
            self.conda_options = CondaOptions(environment_yml_file=conda)
        else:
            self.conda_options = CondaOptions(**conda)

    def save(self, bento_fs: FS):
        # Save all Bento env configs to target Bento directory
        with cwd(self.build_ctx):
            self.docker_options.save(bento_fs)
            self.python_options.save(bento_fs)
            self.conda_options.save(bento_fs)
