import re
import sys
import typing as t
import logging
import platform
from functools import lru_cache

import attr

from ...exceptions import BentoMLException
from ..configuration import is_pypi_installed_bentoml

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
PYTHON_SUPPORTED_VERSIONS = ["3.7", "3.8", "3.9", "3.10"]

# supported CUDA versions
# https://github.com/NVIDIA/cuda-repo-management/issues/4
CUDA_SUPPORTED_VERSIONS = ["11.6", "10.2"]
CUDA_COMPUTE_URL = "https://developer.download.nvidia.com/compute/{task}/repos/{arch}"

CUDA_SUPPORTED_DISTROS = ["debian", "amazonlinux", "ubi8"]
CUDA_URL_UNAME_TARGETARCH_MAPPING = {
        "x86_64": "x86_64",
        "aarch64": "sbsa",
        "ppc64le": "ppc64le",
        }


@lru_cache(maxsize=1)
def get_current_sysarch() -> str:
    # Returns the current CPU architecture of the system.
    return platform.machine()

def get_repo_url() -> t.Tuple[str, str]:
    arch = CUDA_URL_UNAME_TARGETARCH_MAPPING[get_current_sysarch()]
    return CUDA_COMPUTE_URL.format(task='cuda', arch=arch), CUDA_COMPUTE_URL.format(task='machine-learning', arch=arch)

