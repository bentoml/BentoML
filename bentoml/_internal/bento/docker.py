from __future__ import annotations

import os
import re
import typing as t
import logging
from typing import TYPE_CHECKING

import fs
import attr
import yaml
from cattr.gen import override  # type: ignore (incomplete cattr types)
from cattr.gen import make_dict_unstructure_fn  # type: ignore (incomplete cattr types)

from ..utils import bentoml_cattr
from ...exceptions import InvalidArgument
from ...exceptions import BentoMLException

if TYPE_CHECKING:
    P = t.ParamSpec("P")

    from attr import Attribute

logger = logging.getLogger(__name__)

# Python supported versions
DOCKER_SUPPORTED_PYTHON_VERSION = ["3.7", "3.8", "3.9", "3.10"]
# CUDA supported versions
DOCKER_SUPPORTED_CUDA_VERSION = ["11.6.2", "10.2.89"]
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

DOCKER_SUPPORTED_CUDA_DISTROS = [
    i for i in DOCKER_SUPPORTED_DISTRO if DOCKER_SUPPORTED_DISTRO[i][1] is not None
]
DOCKER_SUPPORTED_CONDA_DISTROS = [i for i in DOCKER_SUPPORTED_DISTRO if "conda" in i]


@attr.frozen
class NVIDIALibrary:
    version: str = attr.field(converter=lambda d: "" if d is None else d)
    major: str = attr.field(
        default=attr.Factory(
            lambda self: self.version.split(".")[0] if self.version is not None else "",
            takes_self=True,
        )
    )

    @classmethod
    def from_str(cls, version_str: str) -> NVIDIALibrary:
        return cls(version=version_str)


@attr.frozen
class CUDAVersion:
    major: str
    minor: str
    full: str

    @classmethod
    def from_str(cls, version_str: str) -> CUDAVersion:
        match = re.match(r"^(\d+)\.(\d+)", version_str)
        if match is None:
            raise InvalidArgument(
                f"Invalid CUDA version string: {version_str}. Should follow correct semver format."
            )
        return cls(*match.groups(), version_str)  # type: ignore


def transformer(_: t.Any, fields: list[Attribute[t.Any]]) -> list[Attribute[t.Any]]:
    results: list[Attribute[t.Any]] = []
    for field in fields:
        if field.converter is not None:
            results.append(field)
            continue
        if field.type in {str, "str"}:
            converter = lambda d: d.strip("\n") if isinstance(d, str) else d  # type: ignore
        else:
            converter = None
        results.append(field.evolve(converter=converter))
    return results


def _make_architecture_cuda_cls(arch: str, cuda_spec: dict[str, t.Any]) -> type:
    cls_ = attr.make_class(
        "___cuda_arch_wrapper",
        {
            "requires": attr.attrib(type=str),
            **{
                lib: attr.attrib(type=NVIDIALibrary)
                for lib in cuda_spec[f"components_{arch}"]
            },
        },
        slots=True,
        frozen=True,
        init=True,
        field_transformer=transformer,
    )

    bentoml_cattr.register_unstructure_hook(
        cls_,
        make_dict_unstructure_fn(  # type: ignore
            cls_,
            bentoml_cattr,
            tag=override(omit=True),
        ),
    )
    return cls_


def make_cuda_cls(value: str | None) -> CUDA | None:
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
    except yaml.YAMLError as exc:
        logger.error(exc)
        raise

    architectures = cuda_spec["architectures"]

    cuda_cls = attr.make_class(
        "CUDA",
        {
            "version": attr.attrib(type=CUDAVersion),
            **{
                arch: attr.attrib(
                    type=_make_architecture_cuda_cls(arch, cuda_spec=cuda_spec)
                )
                for arch in architectures
            },
            "repository": attr.attrib(type=str),
        },
        slots=True,
        frozen=True,
        init=True,
    )

    bentoml_cattr.register_unstructure_hook(
        cuda_cls,
        make_dict_unstructure_fn(  # type: ignore
            cuda_cls,
            bentoml_cattr,
            tag=override(omit=True),
        ),
    )

    return cuda_cls(
        version=CUDAVersion.from_str(value),
        repository=f"{cuda_spec['repository']}",
        **{
            arch: _make_architecture_cuda_cls(arch, cuda_spec)(
                requires=cuda_spec[f"requires_{arch}"],
                **{
                    lib: NVIDIALibrary.from_str(lib_spec["version"])
                    for lib, lib_spec in cuda_spec[f"components_{arch}"].items()
                },
            )
            for arch in architectures
        },
    )


def make_distro_cls(value: str) -> Distro:
    if value not in DOCKER_SUPPORTED_DISTRO:
        raise BentoMLException(
            f"{value} is not supported. "
            f"Supported distros are: {', '.join(DOCKER_SUPPORTED_DISTRO.keys())}."
        )
    cls = attr.make_class(
        "Distro",
        {
            "supported_python_version": attr.attrib(
                type=t.List[str],
                validator=attr.validators.deep_iterable(
                    lambda _, __, value: value in DOCKER_SUPPORTED_PYTHON_VERSION,
                    iterable_validator=attr.validators.instance_of(list),
                ),
            ),
            "supported_cuda_version": attr.attrib(
                type=t.Optional[t.List[str]],
                validator=attr.validators.optional(
                    attr.validators.deep_iterable(
                        lambda _, __, value: value in DOCKER_SUPPORTED_CUDA_VERSION,
                    )
                ),
            ),
            "supported_architecture": attr.attrib(
                type=t.List[str],
                validator=attr.validators.deep_iterable(
                    lambda _, __, value: value in DOCKER_SUPPORTED_ARCHITECTURE,
                    iterable_validator=attr.validators.instance_of(list),
                ),
            ),
            "base_image": attr.attrib(type=str),
        },
        slots=True,
        frozen=True,
        init=True,
    )

    setattr(
        cls,
        "from_distro",
        classmethod(lambda cls, value: cls(*DOCKER_SUPPORTED_DISTRO[value])),
    )

    return cls.from_distro(value)  # type: ignore


if TYPE_CHECKING:

    class __cuda_arch_wrapper:
        requires: str
        cudart: NVIDIALibrary
        libcublas: NVIDIALibrary
        libnccl2: NVIDIALibrary
        libcuparse: NVIDIALibrary
        libnpp: NVIDIALibrary
        nvml_dev: NVIDIALibrary
        nvtx: NVIDIALibrary
        nvprof: NVIDIALibrary
        cudnn8: NVIDIALibrary

    class CUDA11x(__cuda_arch_wrapper):
        """CUDA 11.x spec"""

    class CUDA10x(__cuda_arch_wrapper):
        """CUDA 10.x spec"""

        command_line_tools: NVIDIALibrary
        libraries: NVIDIALibrary
        minimal_build: NVIDIALibrary
        nvcc: NVIDIALibrary

    class CUDA:
        repository: str
        version: CUDAVersion
        x86_64: CUDA10x | CUDA11x
        sbsa: CUDA10x | CUDA11x
        ppc64le: CUDA10x | CUDA11x

    class Distro:
        supported_python_version: t.List[str]
        supported_cuda_version: t.Optional[t.List[str]]
        supported_architecture: t.List[str]
        base_image: str

        @classmethod
        def from_distro(cls, value: str) -> Distro:
            ...
