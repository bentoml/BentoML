import typing as t

from ._internal.artifacts import ModelArtifact
from ._internal.types import MetadataType, PathType
from .exceptions import MissingDependencyException

try:
    import fasttext
except ImportError:
    raise MissingDependencyException("fasttext is required by FastTextModel")


class FastTextModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`fasttext` models

    Args:
        model (`fasttext.FastText._FastText`):
            Base pipeline for all fasttext model
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`fasttext` is required by FastTextModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    def __init__(
        self,
        model: "fasttext.FastText._FastText",
        metadata: t.Optional[MetadataType] = None,
    ):
        super(FastTextModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> "fasttext.FastText._FastText":
        return fasttext.load_model(cls.get_path(path))

    def save(self, path: PathType) -> None:
        self._model.save_model(self.get_path(path))
