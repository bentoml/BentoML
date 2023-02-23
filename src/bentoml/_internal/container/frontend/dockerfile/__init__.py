from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING

import attr

from .....exceptions import InvalidArgument
from .....exceptions import BentoMLException

if TYPE_CHECKING:
    P = t.ParamSpec("P")
    ListStr = list[str]
    from ....bento.build_config import CondaOptions
    from ....bento.build_config import DockerOptions
else:
    ListStr = list

logger = logging.getLogger(__name__)


# Python supported versions
SUPPORTED_PYTHON_VERSIONS = ["3.7", "3.8", "3.9", "3.10"]
# CUDA supported versions
SUPPORTED_CUDA_VERSIONS = ["11.7.0", "11.6.2", "11.4.3", "11.2.2"]
# Mapping from user provided version argument to the full version target to install
ALLOWED_CUDA_VERSION_ARGS = {
    "11": "11.7.0",
    "11.7": "11.7.0",
    "11.7.0": "11.7.0",
    "11.6": "11.6.2",
    "11.6.2": "11.6.2",
    "11.4": "11.4.3",
    "11.4.3": "11.4.3",
    "11.2": "11.2.2",
    "11.2.2": "11.2.2",
}

# Supported supported_architectures
SUPPORTED_ARCHITECTURES = ["amd64", "arm64", "ppc64le", "s390x"]
# Supported release types
SUPPORTED_RELEASE_TYPES = ["python", "miniconda", "cuda"]

# BentoML supported distros mapping spec with
# keys represents distros, and value is a tuple of list for supported python
# versions and list of supported CUDA versions.
CONTAINER_METADATA: dict[str, dict[str, t.Any]] = {
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
        "supported_cuda_versions": SUPPORTED_CUDA_VERSIONS,
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
        "supported_cuda_versions": SUPPORTED_CUDA_VERSIONS,
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

CONTAINER_SUPPORTED_DISTROS = list(CONTAINER_METADATA.keys())


def get_supported_spec(spec: t.Literal["python", "miniconda", "cuda"]) -> list[str]:
    if spec not in SUPPORTED_RELEASE_TYPES:
        raise InvalidArgument(
            f"Unknown release type: {spec}, supported spec: {SUPPORTED_RELEASE_TYPES}"
        )
    return [v for v in CONTAINER_METADATA if spec in CONTAINER_METADATA[v]]


@attr.frozen
class DistroSpec:
    name: str = attr.field()
    image: str = attr.field(kw_only=True)

    supported_python_versions: list[str] = attr.field(
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.in_(SUPPORTED_PYTHON_VERSIONS),
            iterable_validator=attr.validators.instance_of(ListStr),
        ),
    )

    supported_cuda_versions: list[str] | None = attr.field(
        default=None,
        validator=attr.validators.optional(
            attr.validators.deep_iterable(
                member_validator=attr.validators.in_(SUPPORTED_CUDA_VERSIONS)
            )
        ),
    )

    supported_architectures: list[str] = attr.field(
        default=SUPPORTED_ARCHITECTURES,
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.in_(SUPPORTED_ARCHITECTURES)
        ),
    )

    @classmethod
    def from_options(cls, docker: DockerOptions, conda: CondaOptions) -> DistroSpec:
        if not docker.distro:
            raise BentoMLException("Distro is required, got None instead.")

        if docker.distro not in CONTAINER_METADATA:
            raise BentoMLException(
                f"{docker.distro} is not supported. Supported distros are: {', '.join(CONTAINER_METADATA.keys())}."
            )

        if docker.cuda_version is not None:
            release_type = "cuda"
        elif not conda.is_empty():
            release_type = "miniconda"
        else:
            release_type = "python"

        meta = CONTAINER_METADATA[docker.distro]

        return cls(
            **meta[release_type],
            name=docker.distro,
            supported_python_versions=meta["supported_python_versions"],
            supported_cuda_versions=meta["supported_cuda_versions"],
        )
