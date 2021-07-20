import typing as t
from pathlib import Path

from ..types import MetadataType, PathType
from ..utils import cloudpickle
from .base import ModelArtifact


class PickleArtifact(ModelArtifact):
    """
    Abstraction for saving/loading python objects with pickle serialization.

    Args:
    model (`Any` that is serializable):
        Data that can be serialized with :obj:`cloudpickle`
    metadata (`Dict[str, Union[Any,...]]`, `optional`, default to `None`):
        dictionary of model metadata
    name (`str`, `optional`, default to `picklemodel`):
        Name of PickleArtifact instance

    .. note::
        We should also provide optional support for using ``pickle``.
        Current caveats when using pickle is that it doesn't work :smile:

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    def __init__(
        self, model: t.Any, metadata: t.Optional[MetadataType] = None,
    ):
        super(PickleArtifact, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> t.Any:
        file: str = cls.model_path(path, cls.PICKLE_FILE_EXTENSION)
        with open(file, "rb") as inf:
            model = cloudpickle.load(inf)
        return model

    def save(self, path: PathType) -> None:
        with open(self.model_path(path, self.PICKLE_FILE_EXTENSION), "wb") as inf:
            cloudpickle.dump(self._model, inf)
