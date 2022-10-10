import logging

from ._internal.frameworks.pytorch_lightning import get
from ._internal.frameworks.pytorch_lightning import load_model
from ._internal.frameworks.pytorch_lightning import save_model
from ._internal.frameworks.pytorch_lightning import get_runnable

logger = logging.getLogger(__name__)


def save(tag, *args, **kwargs):
    logger.warning(
        f'The "{__name__}.save" method is being deprecated. Use "{__name__}.save_model" instead'
    )
    return save_model(tag, *args, **kwargs)


def load(tag, *args, **kwargs):
    logger.warning(
        f'The "{__name__}.load" method is being deprecated. Use "{__name__}.load_model" instead'
    )
    return load_model(tag, *args, **kwargs)


def load_runner(tag, *args, **kwargs):
    if len(args) != 0 or len(kwargs) != 0:
        logger.error(
            f'The "{__name__}.load_runner" method is being deprecated. "load_runner" arguments will be ignored. Use `{__name__}.get("{tag}").to_runner()` instead'
        )
    else:
        logger.warning(
            f'The "{__name__}.load_runner" method is being deprecated. Use `{__name__}.get("{tag}").to_runner()` instead'
        )
    return get(tag).to_runner()


__all__ = ["load_model", "save_model", "get", "get_runnable"]
