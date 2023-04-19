from __future__ import annotations

import typing as t
import logging

from ._internal.frameworks.transformers import get
from ._internal.frameworks.transformers import load_model
from ._internal.frameworks.transformers import save_model
from ._internal.frameworks.transformers import get_runnable
from ._internal.frameworks.transformers import TransformersOptions as ModelOptions

if t.TYPE_CHECKING:
    from ._internal.tag import Tag


logger = logging.getLogger(__name__)


def save(tag: str, *args: t.Any, **kwargs: t.Any):
    logger.warning(
        'The "%s.save" method is deprecated. Use "%s.save_model" instead.',
        __name__,
        __name__,
    )
    return save_model(tag, *args, **kwargs)


def load(tag: Tag | str, *args: t.Any, **kwargs: t.Any):
    logger.warning(
        'The "%s.load" method is deprecated. Use "%s.load_model" instead.',
        __name__,
        __name__,
    )
    return load_model(tag, *args, **kwargs)


def load_runner(tag: Tag | str, *args: t.Any, **kwargs: t.Any):
    if len(args) != 0 or len(kwargs) != 0:
        logger.error(
            'The "%s.load_runner" method is deprecated. "load_runner" arguments will be ignored. Use "%s.get("%s").to_runner()" instead.',
            __name__,
            __name__,
            tag,
        )
    else:
        logger.warning(
            'The "%s.load_runner" method is deprecated. Use "%s.get("%s").to_runner()" instead.',
            __name__,
            __name__,
            tag,
        )
    return get(tag).to_runner()


__all__ = ["load_model", "save_model", "get", "get_runnable", "ModelOptions"]
