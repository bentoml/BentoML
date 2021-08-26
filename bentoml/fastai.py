import os
import typing as t

from ._internal.models.base import MODEL_NAMESPACE, PICKLE_EXTENSION, Model
from ._internal.types import MetadataType, PathType
from ._internal.utils import LazyLoader
from .exceptions import MissingDependencyException

if t.TYPE_CHECKING:
    import fastai  # pylint: disable=unused-import
    import fastai.basics as basics  # pylint: disable=unused-import
    import fastai.learner as learner
else:
    fastai = LazyLoader("fastai", globals(), "fastai")
    basics = LazyLoader("basics", globals(), "fastai.basics")
    learner = LazyLoader("learner", globals(), "fastai.learner")
    assert learner, MissingDependencyException("fastai2 is required by FastAIModel")


class FastAIModel(Model):
    """
    Model class for saving/loading :obj:`fastai` model

    Args:
        model (`fastai.learner.Learner`):
            Learner model from fastai
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
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
        metadata: t.Optional[MetadataType] = None,
    ):
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
