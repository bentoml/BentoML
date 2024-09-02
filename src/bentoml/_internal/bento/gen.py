"""
This module is shim for bentoctl. NOT FOR DIRECT USE.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import fs

from ..container.generate import generate_containerfile
from ..utils.uri import encode_path_for_uri

if TYPE_CHECKING:
    from .build_config import DockerOptions

__all__ = ["generate_dockerfile"]

logger = logging.getLogger(__name__)

warnings.warn(
    "%s is deprecated. Make sure to use 'bentoml.container.build' and 'bentoml.container.health' instead."
    % __name__,
    DeprecationWarning,
    stacklevel=4,
)


def generate_dockerfile(docker: DockerOptions, context_path: str, *, use_conda: bool):
    from ..bento import Bento

    bento = Bento.from_fs(fs.open_fs(encode_path_for_uri(context_path)))
    logger.debug("'use_conda' is deprecated and will not be used.")
    return generate_containerfile(
        docker,
        bento.path,
        conda=bento.info.conda,
        bento_fs=bento._fs,
    )
