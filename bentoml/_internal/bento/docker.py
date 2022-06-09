from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING

import attr

from ...exceptions import InvalidArgument
from ...exceptions import BentoMLException

if TYPE_CHECKING:
    P = t.ParamSpec("P")

logger = logging.getLogger(__name__)


# Python supported versions
SUPPORTED_PYTHON_VERSIONS = ["3.7", "3.8", "3.9", "3.10"]
# CUDA supported versions
SUPPORTED_CUDA_VERSIONS = {"11": "11.6.2"}
# Supported supported_architectures
SUPPORTED_ARCHITECTURES = ["amd64", "arm64", "ppc64le", "s390x"]
# Supported release types
SUPPORTED_RELEASE_TYPES = ["python", "miniconda", "cuda"]

# BentoML supported distros mapping spec with
# keys represents distros, and value is a tuple of list for supported python
# versions and list of supported CUDA versions.
DOCKER_METADATA: dict[str, dict[str, t.Any]] = {
    "amazonlinux": {
        "supported_python_versions": ["3.7", "3.8"],
        "supported_cuda_versions": None,
        "python": {
            "image": "amazonlinux:2",
            "supported_architectures": ["amd64", "arm64"],
        },
    },
    "ubi8": {
        "supported_python_versions": ["3.8", "3.9"],
        "supported_cuda_versions": list(SUPPORTED_CUDA_VERSIONS.values()),
        "python": {
            "image": "registry.access.redhat.com/ubi8/python-{spec_version}:1",
            "supported_architectures": ["amd64", "arm64"],
        },
        "cuda": {
            "image": "nvidia/cuda:{spec_version}-cudnn8-runtime-ubi8",
            "supported_architectures": ["amd64", "arm64", "ppc64le"],
        },
    },
    "debian": {
        "supported_python_versions": SUPPORTED_PYTHON_VERSIONS,
        "supported_cuda_versions": list(SUPPORTED_CUDA_VERSIONS.values()),
        "python": {
            "image": "python:{spec_version}-slim",
            "supported_architectures": SUPPORTED_ARCHITECTURES,
        },
        "cuda": {
            "image": "nvidia/cuda:{spec_version}-cudnn8-runtime-ubuntu20.04",
            "supported_architectures": ["amd64", "arm64"],
        },
        "miniconda": {
            "image": "continuumio/miniconda3:latest",
            "supported_architectures": SUPPORTED_ARCHITECTURES,
        },
    },
    "alpine": {
        "supported_python_versions": SUPPORTED_PYTHON_VERSIONS,
        "supported_cuda_versions": None,
        "python": {
            "image": "python:{spec_version}-alpine",
            "supported_architectures": SUPPORTED_ARCHITECTURES,
        },
        "miniconda": {
            "image": "continuumio/miniconda3:4.10.3p0-alpine",
            "supported_architectures": ["amd64"],
        },
    },
}

DOCKER_SUPPORTED_DISTROS = list(DOCKER_METADATA.keys())


def get_supported_spec(spec: t.Literal["python", "miniconda", "cuda"]) -> list[str]:
    if spec not in SUPPORTED_RELEASE_TYPES:
        raise InvalidArgument(
            f"Unknown release type: {spec}, supported spec: {SUPPORTED_RELEASE_TYPES}"
        )
    return [v for v in DOCKER_METADATA if spec in DOCKER_METADATA[v]]


@attr.define(slots=True, frozen=True)
class DistroSpec:
    name: str = attr.field()
    image: str = attr.field(kw_only=True)
    release_type: str = attr.field(kw_only=True)

    supported_python_versions: t.List[str] = attr.field(
        validator=attr.validators.deep_iterable(
            lambda _, __, value: value in SUPPORTED_PYTHON_VERSIONS,
            iterable_validator=attr.validators.instance_of(list),
        ),
    )

    supported_cuda_versions: t.List[str] = attr.field(
        default=None,
        validator=attr.validators.optional(
            attr.validators.deep_iterable(
                lambda _, __, value: value in SUPPORTED_CUDA_VERSIONS.values(),
            )
        ),
    )

    supported_architectures: t.List[str] = attr.field(
        default=SUPPORTED_ARCHITECTURES,
        validator=attr.validators.deep_iterable(
            lambda _, __, value: value in SUPPORTED_ARCHITECTURES,
        ),
    )

    @classmethod
    def from_distro(
        cls,
        value: str,
        *,
        cuda: bool = False,
        conda: bool = False,
    ) -> DistroSpec:
        if not value:
            raise BentoMLException("Distro name is required, got None instead.")

        if value not in DOCKER_METADATA:
            raise BentoMLException(
                f"{value} is not supported. "
                f"Supported distros are: {', '.join(DOCKER_METADATA.keys())}."
            )

        if cuda:
            release_type = "cuda"
        elif conda:
            release_type = "miniconda"
        else:
            release_type = "python"

        meta = DOCKER_METADATA[value]
        python_version = meta["supported_python_versions"]
        cuda_version = (
            meta["supported_cuda_versions"] if release_type == "cuda" else None
        )

        return cls(
            **meta[release_type],
            name=value,
            release_type=release_type,
            supported_python_versions=python_version,
            supported_cuda_versions=cuda_version,
        )
