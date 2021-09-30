import os
import typing as t

import bentoml._internal.constants as _const

from ._internal.models.base import MODEL_NAMESPACE, PICKLE_EXTENSION, Model
from ._internal.types import GenericDictType, PathType
from ._internal.utils import LazyLoader

_exc = _const.IMPORT_ERROR_MSG.format(
    fwr="scikit-learn",
    module=__name__,
    inst="`pip install scikit-learn`",
)

MT = t.TypeVar("_MT")

try:  # pragma: no cover
    import joblib
except ImportError:  # pragma: no cover
    joblib = LazyLoader("joblib", globals(), "sklearn.externals.joblib", exc_msg=_exc)


class SklearnModel(Model):
    """
    Model class for saving/loading :obj:`sklearn` models.

    Args:
        model (`Any`, that is omitted by `sklearn`):
            Any model that is omitted by `sklearn`
        metadata (`GenericDictType`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`sklearn` is required by SklearnModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento.py`::

        TODO:
    """

    def __init__(self, model: MT, metadata: t.Optional[GenericDictType] = None):
        super(SklearnModel, self).__init__(model, metadata=metadata)

    @staticmethod
    def __get_pickle_fpath(path: PathType) -> PathType:
        return os.path.join(path, f"{MODEL_NAMESPACE}{PICKLE_EXTENSION}")

    @classmethod
    def load(cls, path: PathType) -> t.Any:
        return joblib.load(cls.__get_pickle_fpath(path), mmap_mode="r")

    def save(self, path: PathType) -> None:
        joblib.dump(self._model, self.__get_pickle_fpath(path))
