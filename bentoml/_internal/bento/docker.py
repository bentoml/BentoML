from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING

import attr

from ...exceptions import BentoMLException

if TYPE_CHECKING:
    P = t.ParamSpec("P")

logger = logging.getLogger(__name__)

# Python supported versions
SUPPORTED_PYTHON_VERSION = ["3.7", "3.8", "3.9", "3.10"]
# CUDA supported versions
SUPPORTED_CUDA_VERSION = {"11": "11.6.2"}
# Supported architectures
SUPPORTED_ARCHITECTURE = ["amd64", "arm64", "ppc64le", "s390x"]


# Docker defaults
DEFAULT_CUDA_VERSION = "11.6.2"
DEFAULT_DOCKER_DISTRO = "debian"

# BentoML supported distros mapping spec with
# keys represents distros, and value is a tuple of list for supported python
# versions and list of supported CUDA versions.
DOCKER_SUPPORTED_DISTRO: dict[str, dict[str, t.Any]] = {
    "amazonlinux": {
        "python_version": ["3.8"],
        "cuda_version": None,
        "python": {
            "image": "amazonlinux:2",
            "architecture": ["amd64", "arm64"],
        },
    },
    "ubi8": {
        "python_version": ["3.8", "3.9"],
        "cuda_version": list(SUPPORTED_CUDA_VERSION.values()),
        "python": {
            "image": "registry.access.redhat.com/ubi8/python-{python_version}:1",
            "architecture": ["amd64", "arm64"],
        },
        "cuda": {
            "image": "nvidia/cuda:{cuda_version}-cudnn8-runtime-ubi8",
            "architecture": ["amd64", "arm64", "ppc64le"],
        },
    },
    "debian": {
        "python_version": SUPPORTED_PYTHON_VERSION,
        "cuda_version": list(SUPPORTED_CUDA_VERSION.values()),
        "python": {
            "image": "python:{python_version}-slim",
            "architecture": SUPPORTED_ARCHITECTURE,
        },
        "cuda": {
            "image": "nvidia/cuda:{cuda_version}-cudnn8-runtime-ubuntu20.04",
            "architecture": ["amd64", "arm64"],
        },
    },
    "debian-miniconda": {
        "python_version": SUPPORTED_PYTHON_VERSION,
        "cuda_version": None,
        "miniconda": {
            "image": "mambaorg/micromamba:latest",
            "architecture": ["amd64", "arm64", "ppc64le"],
        },
    },
    "alpine": {
        "python_version": SUPPORTED_PYTHON_VERSION,
        "cuda_version": None,
        "python": {
            "image": "python:{python_version}-alpine",
            "architecture": SUPPORTED_ARCHITECTURE,
        },
    },
    "alpine-miniconda": {
        "python_version": SUPPORTED_PYTHON_VERSION,
        "cuda_version": None,
        "miniconda": {
            "image": "continuumio/miniconda3:4.10.3p0-alpine",
            "architecture": ["amd64"],
        },
    },
}

CUDA_SUPPORTED_DISTRO = [
    v for v in DOCKER_SUPPORTED_DISTRO if "cuda" in DOCKER_SUPPORTED_DISTRO[v]
]
CONDA_SUPPORTED_DISTRO = ["debian-miniconda", "alpine-miniconda"]


@attr.define(slots=True, frozen=True)
class Distro:
    name: str = attr.field()
    image: str = attr.field(kw_only=True)

    python_version: t.List[str] = attr.field(
        validator=attr.validators.deep_iterable(
            lambda _, __, value: value in SUPPORTED_PYTHON_VERSION,
            iterable_validator=attr.validators.instance_of(list),
        ),
    )

    cuda_version: t.List[str] = attr.field(
        default=None,
        validator=attr.validators.optional(
            attr.validators.deep_iterable(
                lambda _, __, value: value in SUPPORTED_CUDA_VERSION.values(),
            )
        ),
    )

    architecture: t.Optional[t.List[str]] = attr.field(
        default=None,
        validator=attr.validators.optional(
            attr.validators.deep_iterable(
                lambda _, __, value: value in SUPPORTED_ARCHITECTURE
            )
        ),
    )

    @classmethod
    def from_distro(
        cls, value: str, docker_release_type: t.Literal["python", "cuda", "miniconda"]
    ) -> Distro:
        if value not in DOCKER_SUPPORTED_DISTRO:
            raise BentoMLException(
                f"{value} is not supported. "
                f"Supported distros are: {', '.join(DOCKER_SUPPORTED_DISTRO.keys())}."
            )

        meta = DOCKER_SUPPORTED_DISTRO[value]
        python_version = meta["python_version"]
        cuda_version = meta["cuda_version"]

        if docker_release_type not in meta:
            raise BentoMLException(
                f"`{value}` does not support {docker_release_type} "
                f"release type. Supported release types are: {', '.join(meta.keys())}."
            )
        return Distro(
            **meta[docker_release_type],
            name=value,
            python_version=python_version,
            cuda_version=cuda_version,
        )
