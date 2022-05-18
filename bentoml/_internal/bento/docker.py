import re
import typing as t
import logging
import platform
from sys import version_info as pyver
from functools import lru_cache

import attr

from ...exceptions import BentoMLException

logger = logging.getLogger(__name__)

# BentoML supported distros
DOCKER_SUPPORTED_DISTROS = [
    "debian",
    "amazonlinux",
    "ubi8",
    "alpine",
    "debian-miniconda",
    "alpine-miniconda",
]

# Python supported versions
SUPPORTED_PYTHON_VERSIONS = ["3.7", "3.8", "3.9", "3.10"]
SUPPORTED_PYTHON_RUNTIME_COMBINATION = {
    "debian": SUPPORTED_PYTHON_VERSIONS,
    "alpine": SUPPORTED_PYTHON_VERSIONS,
    "amazonlinux": ["3.8"],
    "ubi8": ["3.8", "3.9"],
    "debian-miniconda": SUPPORTED_PYTHON_VERSIONS,
    "alpine-miniconda": SUPPORTED_PYTHON_VERSIONS,
}

# supported CUDA versions
# https://github.com/NVIDIA/cuda-repo-management/issues/4
SUPPORTED_CUDA_VERSIONS = ["11.6.1", "10.2"]
CUDA_SUPPORTED_DISTROS = ["debian", "amazonlinux", "ubi8"]

CUDA_COMPUTE_URL = "https://developer.download.nvidia.com/compute/{task}/repos/{arch}"
CUDA_URL_UNAME_TARGETARCH_MAPPING = {
    "x86_64": "x86_64",
    "aarch64": "sbsa",
    "ppc64le": "ppc64le",
}


@lru_cache(maxsize=1)
def get_repo_url() -> t.Tuple[str, str]:
    arch = platform.machine()
    cuda_arch = CUDA_URL_UNAME_TARGETARCH_MAPPING[arch]
    return CUDA_COMPUTE_URL.format(
        task="cuda", arch=cuda_arch
    ), CUDA_COMPUTE_URL.format(task="machine-learning", arch=cuda_arch)
