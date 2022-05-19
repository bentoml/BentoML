from __future__ import annotations

import os
import typing as t
import logging
from typing import TYPE_CHECKING

import fs
import attr
import yaml

from ..utils import bentoml_cattr
from ...exceptions import BentoMLException

if TYPE_CHECKING:
    P = t.ParamSpec("P")
    from .build_config import DockerOptions


logger = logging.getLogger(__name__)


# Python supported versions
DOCKER_SUPPORTED_PYTHON_VERSION = ["3.7", "3.8", "3.9", "3.10"]
# CUDA supported versions
DOCKER_SUPPORTED_CUDA_VERSION = ["11.6.2", "10.2", "10.1"]
# Docker default distros
DOCKER_DEFAULT_DOCKER_DISTRO = "debian"
DOCKER_DEFAULT_CUDA_VERSION = "11.6.2"


@attr.define
class NvidiaLibrary:
    version: str
    major_version: str = attr.field(
        default=attr.Factory(lambda self: self.version.split(".")[0], takes_self=True)
    )


# BentoML supported distros mapping spec with
# keys represents distros, and value is a tuple of list for supported python
# versions and list of supported CUDA versions.
DOCKER_SUPPORTED_DISTRO: t.Dict[str, t.Tuple[t.List[str], t.List[str] | None, str]] = {
    "amazonlinux": (["3.8"], DOCKER_SUPPORTED_CUDA_VERSION, "amazonlinux:2"),
    "ubi8": (
        ["3.8", "3.9"],
        DOCKER_SUPPORTED_CUDA_VERSION,
        "registry.access.redhat.com/ubi8/python-{python_version}:1",
    ),
    "debian": (
        DOCKER_SUPPORTED_PYTHON_VERSION,
        DOCKER_SUPPORTED_CUDA_VERSION,
        "python:{python_version}-slim",
    ),
    "debian-miniconda": (
        DOCKER_SUPPORTED_PYTHON_VERSION,
        DOCKER_SUPPORTED_CUDA_VERSION,
        "mambaorg/micromamba:latest",
    ),
    "alpine": (
        DOCKER_SUPPORTED_PYTHON_VERSION,
        None,
        "python:{python_version}-alpine",
    ),
    "alpine-miniconda": (
        DOCKER_SUPPORTED_PYTHON_VERSION,
        None,
        "continuumio/miniconda3:4.10.3p0-alpine",
    ),
}


def make_cuda_cls(value: str | None) -> type | None:
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

    cls: type = attr.make_class(
        f"_CudaSpecWrapper",
        {
            "requires": attr.attrib(type=str),
            **{lib: attr.attrib(type=NvidiaLibrary) for lib in cuda_spec["components"]},
            "version": attr.attrib(default=value),
        },
        slots=True,
        frozen=True,
        init=True,
    )

    def _cuda_spec_structure_hook(d: t.Any, _: t.Type[object]) -> t.Any:
        update_defaults = {}
        if "components" in d:
            components = d.pop("components")
            update_defaults = {
                lib: NvidiaLibrary(version=lib_spec["version"])
                for lib, lib_spec in components.items()
            }

        return cls(**d, **update_defaults)

    bentoml_cattr.register_structure_hook(cls, _cuda_spec_structure_hook)

    return bentoml_cattr.structure(cuda_spec, cls)


def make_distro_cls(value: str | None) -> type | None:
    if value is None:
        return

    if value not in DOCKER_SUPPORTED_DISTRO:
        raise BentoMLException(
            f"{value} is not supported. Supported distros are: {', '.join(DOCKER_SUPPORTED_DISTRO.keys())}"
        )

    return attr.make_class(
        "_DistroSpecWrapper",
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
                        lambda _, __, value: value in DOCKER_SUPPORTED_CUDA_VERSION
                    )
                ),
            ),
            "base_image": attr.attrib(type=str),
        },
        init=True,
        slots=True,
        frozen=True,
    )(*DOCKER_SUPPORTED_DISTRO[value])


def generate_dockerfile(docker_options: DockerOptions) -> str:
    if docker_options.distro == "debian":
        return generate_debian_dockerfile(docker_options)
    elif docker_options.distro == "alpine":
        return generate_alpine_dockerfile(docker_options)
    elif docker_options.distro == "amazonlinux":
        return generate_amazonlinux_dockerfile(docker_options)
    elif docker_options.distro == "ubi8":
        return generate_ubi8_dockerfile(docker_options)
    elif docker_options.distro == "debian-miniconda":
        return generate_debian_miniconda_dockerfile(docker_options)
    elif docker_options.distro == "alpine-miniconda":
        return generate_alpine_miniconda_dockerfile(docker_options)
    else:
        raise BentoMLException(f"Unsupported distro: {docker_options.distro}")
