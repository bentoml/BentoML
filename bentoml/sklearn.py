import typing as t

from ._internal.artifacts import ModelArtifact
from ._internal.types import MetadataType, PathType
from .exceptions import MissingDependencyException

try:
    import joblib
except ImportError:
    try:
        from sklearn.externals import joblib
    except ImportError:
        raise MissingDependencyException(
            "sklearn module is required to use SklearnModel"
        )


class SklearnModel(ModelArtifact):
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

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    def __init__(self, model, metadata: t.Optional[MetadataType] = None):
        super(SklearnModel, self).__init__(model, metadata=metadata)

    @classmethod
    def __get_file_path(cls, path: PathType) -> PathType:
        return cls.get_path(path, cls.PICKLE_EXTENSION)

    @classmethod
    def load(cls, path: PathType) -> t.Any:
        return joblib.load(cls.__get_file_path(path), mmap_mode="r")

    def save(self, path: PathType) -> None:
        joblib.dump(self._model, self.__get_file_path(path))
