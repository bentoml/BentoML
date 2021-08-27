import os
import typing as t

from ._internal.models.base import MODEL_NAMESPACE, PICKLE_EXTENSION, Model
from ._internal.types import MetadataType, PathType
from ._internal.utils import LazyLoader

MT = t.TypeVar("MT")

try:
    import joblib
except ImportError:
    joblib = LazyLoader("joblib", globals(), "sklearn.externals.joblib")


class SklearnModel(Model):
    """
    Model class for saving/loading :obj:`sklearn` models.

    Args:
        model (`Any`, that is omitted by `sklearn`):
            Any model that is omitted by `sklearn`
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`sklearn` is required by SklearnModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento.py`::

        TODO:
    """

    def __init__(self, model: MT, metadata: t.Optional[MetadataType] = None):
        super(SklearnModel, self).__init__(model, metadata=metadata)

    @staticmethod
    def __get_pickle_fpath(path: PathType) -> PathType:
        return os.path.join(path, f"{MODEL_NAMESPACE}{PICKLE_EXTENSION}")

    @classmethod
    def load(cls, path: PathType) -> t.Any:
        return joblib.load(cls.__get_pickle_fpath(path), mmap_mode="r")

    def save(self, path: PathType) -> None:
        joblib.dump(self._model, self.__get_pickle_fpath(path))
