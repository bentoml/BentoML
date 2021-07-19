import typing as t
from pathlib import Path

from ..types import PathType
from ..utils import cloudpickle
from .base import BaseArtifact


class PickleArtifact(BaseArtifact):
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
        Current caveats when using pickle is that it doesn't work =)

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    def __init__(
        self,
        model: t.Any,
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
        name: t.Optional[str] = "picklemodel",
    ):
        super(PickleArtifact, self).__init__(model, metadata=metadata, name=name)

    @classmethod
    def load(cls, path: PathType) -> t.Any:
        f: Path = cls.get_path(path, cls.PICKLE_FILE_EXTENSION)
        with f.open("rb") as inf:
            model = cloudpickle.load(inf)
        return model

    def save(self, path: PathType) -> None:
        with open(self.model_path(path, self.PICKLE_FILE_EXTENSION), "wb") as inf:
            cloudpickle.dump(self._model, inf)
