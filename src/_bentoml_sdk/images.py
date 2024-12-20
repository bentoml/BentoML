from __future__ import annotations

import logging
import platform
import subprocess
import sys
import typing as t
from pathlib import Path

import attrs

from bentoml._internal.bento.bento import ImageInfo
from bentoml._internal.bento.build_config import BentoBuildConfig
from bentoml._internal.configuration import get_bentoml_requirement
from bentoml._internal.configuration import get_debug_mode
from bentoml._internal.configuration import get_quiet_mode
from bentoml._internal.container.frontend.dockerfile import CONTAINER_METADATA
from bentoml._internal.container.frontend.dockerfile import CONTAINER_SUPPORTED_DISTROS
from bentoml._internal.utils.pkg import get_local_bentoml_dependency
from bentoml.exceptions import BentoMLConfigException
from bentoml.exceptions import BentoMLException

logger = logging.getLogger("bentoml.build")

DEFAULT_PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"


@attrs.define
class Image:
    """A class defining the environment requirements for bento."""

    base_image: str
    python_version: str = DEFAULT_PYTHON_VERSION
    commands: t.List[str] = attrs.field(factory=list)
    lock_python_packages: bool = True
    python_requirements: str = ""
    post_commands: t.List[str] = attrs.field(factory=list)
    _after_pip_install: bool = attrs.field(init=False, default=False, repr=False)

    def requirements_file(self, file_path: str) -> t.Self:
        """Add a requirements file to the image. Supports chaining call.

        Example:

        .. code-block:: python

            image = Image("debian:latest").requirements_file("requirements.txt")
        """
        self.python_requirements += Path(file_path).read_text()
        self._after_pip_install = True
        return self

    def python_packages(self, *packages: str) -> t.Self:
        """Add python dependencies to the image. Supports chaining call.

        Example:

        .. code-block:: python

            image = Image("debian:latest")\
                .python_packages("numpy", "pandas")\
                .requirements_file("requirements.txt")
        """
        self.python_requirements += "\n".join(packages)
        self._after_pip_install = True
        return self

    def run(self, command: str) -> t.Self:
        """Add a command to the image. Supports chaining call.

        Example:

        .. code-block:: python

            image = Image("debian:latest").run("echo 'Hello, World!'")
        """
        commands = self.post_commands if self._after_pip_install else self.commands
        commands.append(command)
        return self

    def freeze(self, platform_: str | None = None) -> ImageInfo:
        """Freeze the image to an ImageInfo object for build."""
        if not self.lock_python_packages:
            python_requirements = self.python_requirements
        else:
            python_requirements = self._freeze_python_requirements(platform_)
        return ImageInfo(
            base_image=self.base_image,
            python_version=self.python_version,
            commands=self.commands,
            python_requirements=python_requirements,
            post_commands=self.post_commands,
        )

    def _freeze_python_requirements(self, platform_: str | None = None) -> str:
        from tempfile import TemporaryDirectory

        from pip_requirements_parser import RequirementsFile

        with TemporaryDirectory(prefix="bento-reqs-") as parent:
            requirements_in = Path(parent).joinpath("requirements.in")
            requirements_in.write_text(self.python_requirements)
            # XXX: RequirementsFile.from_string() does not work due to bugs
            requirements_file = RequirementsFile.from_file(str(requirements_in))
            has_bentoml_req = any(
                req.name and req.name.lower() == "bentoml"
                for req in requirements_file.requirements
            )
            with requirements_in.open("w") as f:
                f.write(requirements_file.dumps(preserve_one_empty_line=True))
                if not has_bentoml_req:
                    req = get_bentoml_requirement() or get_local_bentoml_dependency()
                    f.write(f"{req}\n")
            lock_args = [
                str(requirements_in),
                "--allow-unsafe",
                "--no-header",
                f"--output-file={requirements_in.with_suffix('.lock')}",
                "--emit-index-url",
                "--emit-find-links",
                "--no-annotate",
            ]
            if get_debug_mode():
                lock_args.append("--verbose")
            else:
                lock_args.append("--quiet")
            logger.info("Locking PyPI package versions.")
            if platform_:
                lock_args.extend(["--python-platform", platform_])
            elif platform.system() != "Linux" or platform.machine() != "x86_64":
                logger.info(
                    "Locking packages for x86_64-unknown-linux-gnu. "
                    "Pass `--platform` option to specify the platform."
                )
                lock_args.extend(["--python-platform", "linux"])
            cmd = [sys.executable, "-m", "uv", "pip", "compile", *lock_args]
            try:
                subprocess.check_call(
                    cmd,
                    text=True,
                    stderr=subprocess.DEVNULL if get_quiet_mode() else None,
                    cwd=parent,
                )
            except subprocess.CalledProcessError as e:
                raise BentoMLException(f"Failed to lock PyPI packages: {e}") from None
            return requirements_in.with_suffix(".lock").read_text()


@attrs.define
class PythonImage(Image):
    base_image: str = ""
    distro: str = "debian"
    _original_base_image: str = attrs.field(init=False, default="")

    def __attrs_post_init__(self) -> None:
        self._original_base_image = self.base_image
        if not self.base_image:
            if self.distro not in CONTAINER_METADATA:
                raise BentoMLConfigException(
                    f"Unsupported distro: {self.distro}, expected one of {CONTAINER_SUPPORTED_DISTROS}"
                )
            python_version = self.python_version
            if self.distro in ("ubi8",):
                python_version = python_version.replace(".", "")
            distro_config = CONTAINER_METADATA[self.distro]
            self.base_image = distro_config["python"]["image"].format(
                spec_version=python_version
            )
            self.commands.insert(0, distro_config["default_install_command"])

    def system_packages(self, *packages: str) -> t.Self:
        if self._original_base_image:
            raise BentoMLConfigException(
                "system_packages() can only be used in default base image"
            )
        if self.distro not in CONTAINER_METADATA:
            raise BentoMLConfigException(
                f"Unsupported distro: {self.distro}, expected one of {CONTAINER_SUPPORTED_DISTROS}"
            )
        self.commands.append(
            CONTAINER_METADATA[self.distro]["install_command"].format(
                packages=" ".join(packages)
            )
        )
        return self


def get_image_from_build_config(build_config: BentoBuildConfig) -> Image | None:
    if not build_config.conda.is_empty():
        logger.warning(
            "conda options are not supported by bento v2, fallback to bento v1"
        )
        return None
    image = PythonImage()
    docker_options = build_config.docker
    if docker_options.cuda_version is not None:
        logger.warning(
            "docker.cuda_version is not supported by bento v2, fallback to bento v1"
        )
        return None
    if docker_options.dockerfile_template is not None:
        logger.warning(
            "docker.dockerfile_template is not supported by bento v2, fallback to bento v1"
        )
        return None
    if docker_options.setup_script is not None:
        logger.warning(
            "docker.setup_script is not supported by bento v2, fallback to bento v1"
        )
        return None
    if docker_options.base_image is not None:
        image.base_image = docker_options.base_image
    if docker_options.distro is not None:
        image.distro = docker_options.distro
    if docker_options.python_version is not None:
        image.python_version = docker_options.python_version
    if docker_options.system_packages:
        image.system_packages(*docker_options.system_packages)

    python_options = build_config.python.with_defaults()
    if python_options.wheels:
        logger.warning(
            "python.wheels is not supported by bento v2, fallback to bento v1"
        )
        return None
    image.lock_python_packages = python_options.lock_packages
    if python_options.index_url:
        image.python_packages(f"--index-url {python_options.index_url}")
    if python_options.no_index:
        image.python_packages("--no-index")
    if python_options.trusted_host:
        image.python_packages(
            *(f"--trusted-host {h}" for h in python_options.trusted_host)
        )
    if python_options.extra_index_url:
        image.python_packages(
            *(f"--extra-index-url {url}" for url in python_options.extra_index_url)
        )
    if python_options.find_links:
        image.python_packages(
            *(f"--find-links {link}" for link in python_options.find_links)
        )
    if python_options.requirements_txt:
        image.requirements_file(python_options.requirements_txt)
    if python_options.packages:
        image.python_packages(*python_options.packages)
    return image
