from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import re
import typing as t
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
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


def extract_commit_hash(
    resolved_dir: str, regex_commit_hash: t.Pattern[str]
) -> str | None:
    """
    Extracts the commit hash from a resolved filename toward a cache file.
    modified from https://github.com/huggingface/transformers/blob/0b7b4429c78de68acaf72224eb6dae43616d820c/src/transformers/utils/hub.py#L219
    """

    resolved_dir = str(Path(resolved_dir).as_posix()) + "/"
    search = re.search(r"snapshots/([^/]+)/", resolved_dir)

    if search is None:
        return None

    commit_hash = search.groups()[0]
    return commit_hash if regex_commit_hash.match(commit_hash) else None


@contextmanager
def init_empty_weights(include_buffers: bool = False):
    """
    init_empty_weights is vendored from accelerate. This is a useful function as it allows us to call transformers.AutoModel.from_pretrained without loading
    the actual weights of the model. This grants access to the model class and its attributes without using a significant amount of memory.

    A context manager under which models are initialized with all parameters on the meta device, therefore creating an
    empty model. Useful when just initializing the model would blow the available RAM.

    Args:
        include_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```pyton
    import torch.nn as nn
    from accelerate import init_empty_weights

    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```

    <Tip warning={true}>

    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].

    </Tip>
    """
    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            module._parameters[name] = param_cls(
                module._parameters[name].to(torch.device("meta")), **kwargs
            )

    def register_empty_buffer(module, name, buffer):
        old_register_buffer(module, name, buffer)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(torch.device("meta"))

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer
