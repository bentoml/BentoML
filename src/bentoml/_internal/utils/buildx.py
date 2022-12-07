"""
This module is shim for bentoctl. NOT FOR DIRECT USE.
Make sure to use 'bentoml.container.build' and 'bentoml.container.health' instead.
"""

from __future__ import annotations

import typing as t
import warnings

from ..container import health as _internal_container_health
from ..container import get_backend

__all__ = ["build", "health"]

_buildx_backend = get_backend("buildx")

warnings.warn(
    "%s is deprecated. Make sure to use 'bentoml.container.build' and 'bentoml.container.health' instead."
    % __name__,
    DeprecationWarning,
    stacklevel=4,
)


def health():
    return _internal_container_health("buildx")


def build(**kwargs: t.Any):
    # subprocess_env from bentoctl will be handle by buildx, so it is safe to pop this out.
    kwargs.pop("subprocess_env")
    kwargs["tag"] = kwargs.pop("tags")
    context_path = kwargs.pop("cwd", None)
    for key, value in kwargs.items():
        if not value:
            kwargs[key] = None
    _buildx_backend.build(context_path=context_path, **kwargs)
