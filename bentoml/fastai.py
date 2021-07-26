import os
import typing as t

from ._internal.models.base import MODEL_NAMESPACE, PICKLE_EXTENSION, Model
from ._internal.types import MetadataType, PathType
from .exceptions import MissingDependencyException

try:
    import fastai
    import fastai.basics  # noqa

    # fastai v2
    import fastai.learner
except ImportError:
    raise MissingDependencyException("fastai v2 is required by FastAIModel")


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

    def __init__(
        self,
        model: "fastai.learner.Learner",
        metadata: t.Optional[MetadataType] = None,
    ):
        super(FastAIModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> "fastai.learner.Learner":
        return fastai.basics.load_learner(
            os.path.join(path, f"{MODEL_NAMESPACE}{PICKLE_EXTENSION}")
        )

    def save(self, path: PathType) -> None:
        self._model.export(
            fname=os.path.join(path, f"{MODEL_NAMESPACE}{PICKLE_EXTENSION}")
        )
