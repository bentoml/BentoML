import logging

from ._internal.frameworks.fastai import get
from ._internal.frameworks.fastai import load_model
from ._internal.frameworks.fastai import save_model
from ._internal.frameworks.fastai import get_runnable
from ._internal.frameworks.fastai import FastAIOptions as ModelOptions  # type: ignore # noqa

logger = logging.getLogger(__name__)


__all__ = ["load_model", "save_model", "get", "get_runnable"]
