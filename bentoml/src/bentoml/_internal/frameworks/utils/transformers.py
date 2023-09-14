from __future__ import annotations

import importlib.metadata
import importlib.util
import logging

from packaging import version

logger = logging.getLogger(__name__)


def is_flax_available():
    _flax_available = (
        importlib.util.find_spec("jax") is not None
        and importlib.util.find_spec("flax") is not None
    )
    if _flax_available:
        try:
            importlib.metadata.version("jax")
            importlib.metadata.version("flax")
        except importlib.metadata.PackageNotFoundError:
            _flax_available = False
    return _flax_available


def is_torch_available():
    _torch_available = importlib.util.find_spec("torch") is not None
    if _torch_available:
        try:
            importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            _torch_available = False
    return _torch_available


def is_tf_available():
    from .tensorflow import get_tf_version

    _tf_available = importlib.util.find_spec("tensorflow") is not None
    if _tf_available:
        _tf_version = get_tf_version()
        _tf_available = _tf_version != ""
        if version.parse(_tf_version) < version.parse("2"):
            logger.info(
                f"Tensorflow found but with verison {_tf_version}. Transformers support with BentoML requires a minimum of Tensorflow 2 and above."
            )
            _tf_available = False
    return _tf_available
