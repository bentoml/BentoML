from __future__ import annotations

import os
import re
import typing as t
import logging
from typing import TYPE_CHECKING

import fs
import attr
import yaml

from ..utils import bentoml_cattr
from ...exceptions import InvalidArgument
from ...exceptions import BentoMLException

if TYPE_CHECKING:
    P = t.ParamSpec("P")


logger = logging.getLogger(__name__)

# Python supported versions
DOCKER_SUPPORTED_PYTHON_VERSION = ["3.7", "3.8", "3.9", "3.10"]
# CUDA supported versions
DOCKER_SUPPORTED_CUDA_VERSION = ["11.6.2", "10.2"]
# Supported architectures
DOCKER_SUPPORTED_ARCHITECTURE = ["amd64", "arm64", "ppc64le", "s390x"]


# Docker defaults
DOCKER_DEFAULT_CUDA_VERSION = "11.6.2"
DOCKER_DEFAULT_DOCKER_DISTRO = "debian"

# BentoML supported distros mapping spec with
# keys represents distros, and value is a tuple of list for supported python
# versions and list of supported CUDA versions.
DOCKER_SUPPORTED_DISTRO: t.Dict[
    str, t.Tuple[t.List[str], t.List[str] | None, t.List[str], str]
] = {
    "amazonlinux": (
        ["3.8"],
        DOCKER_SUPPORTED_CUDA_VERSION,
        ["amd64", "arm64"],
        "amazonlinux:2",
    ),
    "ubi8": (
        ["3.8", "3.9"],
        DOCKER_SUPPORTED_CUDA_VERSION,
        ["amd64", "arm64"],
        "registry.access.redhat.com/ubi8/python-{python_version}:1",
    ),
    "debian": (
        DOCKER_SUPPORTED_PYTHON_VERSION,
        DOCKER_SUPPORTED_CUDA_VERSION,
        DOCKER_SUPPORTED_ARCHITECTURE,
        "python:{python_version}-slim",
    ),
    "debian-miniconda": (
        DOCKER_SUPPORTED_PYTHON_VERSION,
        DOCKER_SUPPORTED_CUDA_VERSION,
        ["amd64", "arm64", "ppc64le"],
        "mambaorg/micromamba:latest",
    ),
    "alpine": (
        DOCKER_SUPPORTED_PYTHON_VERSION,
        None,
        DOCKER_SUPPORTED_ARCHITECTURE,
        "python:{python_version}-alpine",
    ),
    "alpine-miniconda": (
        DOCKER_SUPPORTED_PYTHON_VERSION,
        None,
        ["amd64"],
        "continuumio/miniconda3:4.10.3p0-alpine",
    ),
}

DOCKER_SUPPORTED_CUDA_DISTRO = [
    i for i in DOCKER_SUPPORTED_DISTRO if DOCKER_SUPPORTED_DISTRO[i][1] is not None
]


@attr.define(frozen=True)
class NVIDIALibrary:
    version: str
    major_version: str = attr.field(
        default=attr.Factory(lambda self: self.version.split(".")[0], takes_self=True)
    )


@attr.define(frozen=True)
class CUDAVersion:
    major: str
    minor: str
    full: str

    @classmethod
    def from_str(cls, version_str: str) -> CUDAVersion:
        match = re.match(r"^(\d+)\.(\d+)$", version_str)
        if match is None:
            raise InvalidArgument(
                f"Invalid CUDA version string: {version_str}. Should follow correct semver format."
            )
        return cls(*match.groups(), version_str)  # type: ignore


def make_cuda_cls(value: str | None) -> _CUDASpec10Type | _CUDASpec11Type | None:
    if value is None:
        return

    if value not in DOCKER_SUPPORTED_CUDA_VERSION:
        raise BentoMLException(
            f"CUDA version {value} is not supported. Supported versions: {', '.join(DOCKER_SUPPORTED_CUDA_VERSION)}"
        )

    cuda_folder = fs.path.join(os.path.dirname(__file__), "docker", "cuda")
    cuda_file = fs.path.combine(cuda_folder, f"v{value}.yaml")

    try:
        with open(cuda_file, "r", encoding="utf-8") as f:
            cuda_spec = yaml.safe_load(f.read())
    except FileNotFoundError as fs_exc:
        raise BentoMLException(
            f"{value} is defined in DOCKER_SUPPORTED_CUDA_VERSION but {cuda_file} is not found."
        ) from fs_exc
    except yaml.YAMLError as exc:
        logger.error(exc)
        raise

    if "requires" not in cuda_spec:
        raise BentoMLException(
            "Missing 'requires' key in given CUDA version definition. "
            "Most likely this is an internal mistake when defining "
            f"supported CUDA versions under {cuda_folder}"
        )

    cls = attr.make_class(
        "_CUDASpecWrapper",
        {
            "requires": attr.attrib(type=str),
            **{lib: attr.attrib(type=NVIDIALibrary) for lib in cuda_spec["components"]},
            "version": attr.attrib(type=CUDAVersion),
        },
        slots=True,
        frozen=True,
        init=True,
    )

    def _cuda_spec_structure_hook(
        d: t.Any, _: t.Type[t.Any]
    ) -> t.Union[_CUDASpec10Type, _CUDASpec11Type]:
        update_spec = {}
        if "components" in d:
            components = d.pop("components")
            update_spec = {
                lib: NVIDIALibrary(version=lib_spec["version"])
                for lib, lib_spec in components.items()
            }

        return cls(**d, **update_spec, version=CUDAVersion.from_str(value))

    bentoml_cattr.register_structure_hook(cls, _cuda_spec_structure_hook)

    return bentoml_cattr.structure(cuda_spec, cls)


@attr.define(frozen=True, slots=True, on_setattr=None)
class _DistroSpecWrapper:
    supported_python_version: t.List[str] = attr.field(
        validator=attr.validators.deep_iterable(
            lambda _, __, value: value in DOCKER_SUPPORTED_PYTHON_VERSION,
            iterable_validator=attr.validators.instance_of(list),
        ),
    )
    supported_cuda_version: t.Optional[t.List[str]] = attr.field(
        validator=attr.validators.optional(
            attr.validators.deep_iterable(
                lambda _, __, value: value in DOCKER_SUPPORTED_CUDA_VERSION
            )
        ),
    )
    supported_architecture: t.List[str] = attr.field(
        validator=attr.validators.deep_iterable(
            lambda _, __, value: value in DOCKER_SUPPORTED_ARCHITECTURE,
            iterable_validator=attr.validators.instance_of(list),
        ),
    )

    base_image: str


def make_distro_cls(value: str | None) -> _DistroSpecWrapper | None:
    if value is None:
        return

    if value not in DOCKER_SUPPORTED_DISTRO:
        raise BentoMLException(
            f"{value} is not supported. Supported distros are: {', '.join(DOCKER_SUPPORTED_DISTRO.keys())}"
        )

    return _DistroSpecWrapper(*DOCKER_SUPPORTED_DISTRO[value])


if TYPE_CHECKING:

    class _CUDASpec11Type:
        requires: str
        cudart: NVIDIALibrary
        libcublas: NVIDIALibrary
        libcuparse: NVIDIALibrary
        libnpp: NVIDIALibrary
        nvml_dev: NVIDIALibrary
        nvprof: NVIDIALibrary
        nvtx: NVIDIALibrary
        libnccl2: NVIDIALibrary
        cudnn8: NVIDIALibrary
        version: CUDAVersion

    class _CUDASpec10Type:
        requires: str
        cudart: NVIDIALibrary
        nvml_dev: NVIDIALibrary
        command_line_tools: NVIDIALibrary
        libcusparse: NVIDIALibrary
        libnpp: NVIDIALibrary
        libraries: NVIDIALibrary
        minimal_build: NVIDIALibrary
        nvtx: NVIDIALibrary
        nvprof: NVIDIALibrary
        nvcc: NVIDIALibrary
        libcublas: NVIDIALibrary
        libnccl2: NVIDIALibrary
        cudnn8: NVIDIALibrary
        version: CUDAVersion
