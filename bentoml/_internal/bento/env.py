import logging
import os
import re
import typing as t
from contextlib import contextmanager
from sys import version_info as pyver

import attr
import fs
import fs.copy
from fs.base import FS

from ...exceptions import InvalidArgument
from .docker import ImageProvider
from .templates import BENTO_SERVER_DOCKERFILE

logger = logging.getLogger(__name__)

PYTHON_VERSION: str = f"{pyver.major}.{pyver.minor}.{pyver.micro}"
PYTHON_MINOR_VERSION: str = f"{pyver.major}.{pyver.minor}"
PYTHON_SUPPORTED_VERSIONS: t.List[str] = ["3.7", "3.8", "3.9", "3.10"]
DOCKER_SUPPORTED_DISTROS: t.List[str] = [
    "slim",
    "amazonlinux2",
    "alpine",
    "centos7",
    "centos8",
]
DOCKER_DEFAULT_DISTRO = "slim"

if PYTHON_MINOR_VERSION not in PYTHON_SUPPORTED_VERSIONS:
    logger.warning(
        "BentoML may not work well with current python version %, "
        "supported python versions are: %",
        PYTHON_MINOR_VERSION,
        ",".join(PYTHON_SUPPORTED_VERSIONS),
    )


@contextmanager
def cwd(path: str):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


def _convert_python_version(py_version: str) -> str:
    match = re.match(r"^(\d+).(\d+)", py_version)
    if match is None:
        raise InvalidArgument(
            f'Invalid build option: docker.python_version="{py_version}", python '
            f"version must follow standard python semver format, e.g. 3.7.10 ",
        )
    major, minor = match.groups()
    return f"{major}.{minor}"


@attr.define(frozen=True)
class DockerOptions:
    # Options for choosing a BentoML built-in docker images
    distro: str = attr.ib(
        validator=attr.validators.in_(DOCKER_SUPPORTED_DISTROS),
        default=DOCKER_DEFAULT_DISTRO,
    )
    python_version: str = attr.ib(
        converter=_convert_python_version,
        default=PYTHON_MINOR_VERSION,
        validator=attr.validators.in_(PYTHON_SUPPORTED_VERSIONS),
    )
    gpu: bool = attr.ib(default=False)

    # A python or shell script that executes during docker build time
    setup_script: t.Optional[str] = None

    # A user-provided custom docker image
    base_image: t.Optional[str] = None

    def __attrs_post_init__(self):
        if self.base_image is not None:
            if self.distro is not None:
                logger.warning(
                    "docker base_image %s is used, 'distro=%s' option is ignored",
                    self.base_image,
                    self.distro,
                )
            if self.python_version is not None:
                logger.warning(
                    "docker base_image %s is used, 'python=%s' option is ignored",
                    self.base_image,
                    self.python_version,
                )
            if self.gpu is not None:
                logger.warning(
                    "docker base_image %s is used, 'gpu=%s' option is ignored",
                    self.base_image,
                    self.gpu,
                )

    def get_base_image_tag(self):
        if self.base_image is None:
            # TODO: remove this after 1.0 images published
            # Override to a fixed image for development purpose
            base_image = repr(ImageProvider(self.distro, self.python_version, self.gpu))
            base_image = "bentoml/model-server:0.13.1-slim-py37"
            return base_image
        else:
            return self.base_image

    def write_to_bento(self, bento_fs: FS):
        docker_folder = fs.path.join("env", "docker")
        bento_fs.makedirs(docker_folder, recreate=True)
        dockerfile = fs.path.join(docker_folder, "Dockerfile")
        with bento_fs.open(dockerfile, "w") as dockerfile:
            dockerfile.write(
                BENTO_SERVER_DOCKERFILE.format(base_image=self.get_base_image_tag())
            )

        current_dir = fs.open_fs(os.path.dirname(__file__))
        for filename in ["bentoml-init.sh", "docker-entrypoint.sh"]:
            fs.copy.copy_file(
                current_dir, filename, bento_fs, fs.path.join(docker_folder, filename)
            )

        # TODO: copy over self.setup_script


@attr.define(frozen=True)
class CondaOptions:
    environment_yml: t.Optional[str] = None
    channels: t.Optional[t.List[str]] = None
    dependencies: t.Optional[t.List[str]] = None
    pip: t.Optional[t.List[str]] = None  # list of pip packages to install via conda

    def write_to_bento(self, bento_fs: FS):
        ...


@attr.define(frozen=True)
class PythonOptions:
    packages: t.Optional[t.List[str]]
    index_url: t.Optional[str] = None
    no_index: bool = False
    trusted_host: t.Optional[t.List[str]] = None
    find_links: t.Optional[t.List[str]] = None
    extra_index_url: t.Optional[t.List[str]] = None
    pip_args: t.Optional[str] = None
    wheels: t.List[str] = attr.ib(factory=list)

    def write_to_bento(self, bento_fs: FS):
        ...
