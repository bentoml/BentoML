from __future__ import annotations

import sys
import typing as t
from pathlib import Path

import attrs

from bentoml._internal.bento.bento import ImageInfo
from bentoml._internal.container.frontend.dockerfile import CONTAINER_METADATA
from bentoml._internal.container.frontend.dockerfile import CONTAINER_SUPPORTED_DISTROS
from bentoml.exceptions import BentoMLConfigException

DEFAULT_PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"


@attrs.define
class Image:
    """A class defining the environment requirements for bento."""

    base_image: str
    python_version: str = DEFAULT_PYTHON_VERSION
    commands: t.List[str] = attrs.field(factory=list)
    python_requirements: str = ""

    def requirements_file(self, file_path: str) -> t.Self:
        """Add a requirements file to the image. Supports chaining call.

        Example:

        .. code-block:: python

            image = Image("debian:latest").requirements_file("requirements.txt")
        """
        self.python_requirements += Path(file_path).read_text()
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
        return self

    def run(self, command: str) -> t.Self:
        """Add a command to the image. Supports chaining call.

        Example:

        .. code-block:: python

            image = Image("debian:latest").run("echo 'Hello, World!'")
        """
        self.commands.append(command)
        return self

    def freeze(self) -> ImageInfo:
        """Freeze the image to an ImageInfo object for build."""
        return ImageInfo(
            base_image=self.base_image,
            python_version=self.python_version,
            commands=self.commands,
            python_requirements=self.python_requirements,
        )


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

            self.base_image = CONTAINER_METADATA[self.distro]["python"]["image"].format(
                spec_version=self.python_version
            )

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
