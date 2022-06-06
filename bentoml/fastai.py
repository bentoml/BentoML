from __future__ import annotations

import logging
from typing import Any
from typing import TYPE_CHECKING

from ._internal.frameworks.fastai import get
from ._internal.frameworks.fastai import load_model
from ._internal.frameworks.fastai import save_model
from ._internal.frameworks.fastai import get_runnable

if TYPE_CHECKING:

    from .models import Model
    from ._internal.tag import Tag
    from ._internal.runner import Runner
    from ._internal.frameworks.fastai import Learner

logger = logging.getLogger(__name__)


def save(tag: str, *args: Any, **kwargs: Any) -> Tag:
    logger.warning(
        f'The "{__name__}.save" method is being deprecated. Use "{__name__}.save_model" instead'
    )
    return save_model(tag, *args, **kwargs)


def load(tag: str | Tag | Model, *args: Any, **kwargs: Any) -> Learner:
    logger.warning(
        f'The "{__name__}.load" method is being deprecated. Use "{__name__}.load_model" instead'
    )
    return load_model(tag, *args, **kwargs)


def load_runner(tag: str | Tag, *args: Any, **kwargs: Any) -> Runner:
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
