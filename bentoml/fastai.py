import os
import typing as t
from typing import TYPE_CHECKING

import bentoml._internal.constants as _const

from ._internal.models.base import MODEL_NAMESPACE, PICKLE_EXTENSION, Model
from ._internal.types import GenericDictType, PathType
from ._internal.utils import LazyLoader
from .exceptions import BentoMLException

_exc = _const.IMPORT_ERROR_MSG.format(
    fwr="fastai",
    module=__name__,
    inst="Make sure to install PyTorch first,"
    " then `pip install fastai`. Refers to"
    " https://github.com/fastai/fastai#installing",
)
if TYPE_CHECKING:  # pragma: no cover
    import fastai
    import fastai.basics as basics
    import fastai.learner as learner
else:
    fastai = LazyLoader("fastai", globals(), "fastai", exc_msg=_exc)
    basics = LazyLoader("basics", globals(), "fastai.basics", exc_msg=_exc)
    learner = LazyLoader("learner", globals(), "fastai.learner", exc_msg=_exc)


class FastAIModel(Model):
    """
    Model class for saving/loading :obj:`fastai` model

    Args:
        model (`fastai.learner.Learner`):
            Learner model from fastai
        metadata (`GenericDictType`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`fastai` is required by FastAIModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento.py`::

        TODO:
    """

    _model: "learner.Learner"

    def __init__(
        self,
        model: "learner.Learner",
        metadata: t.Optional[GenericDictType] = None,
    ):
        assert learner, BentoMLException("Only fastai2 is supported by BentoML")
        super(FastAIModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> "learner.Learner":
        return basics.load_learner(
            os.path.join(path, f"{MODEL_NAMESPACE}{PICKLE_EXTENSION}")
        )

    def save(self, path: PathType) -> None:
        self._model.export(
            fname=os.path.join(path, f"{MODEL_NAMESPACE}{PICKLE_EXTENSION}")
        )
