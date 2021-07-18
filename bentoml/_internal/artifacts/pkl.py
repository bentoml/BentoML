import typing as t
from pathlib import Path

from ..types import MT, PathType
from ..utils import cloudpickle
from .base import BaseArtifact


class PickleArtifact(BaseArtifact):
    """
    Abstraction for saving/loading python objects with pickle serialization.

    Class attributes:

    - model (`Any` that is serializable):
        Data that can be serialized with :obj:`cloudpickle`
    - metadata (`Dict[str, Union[Any,...]]`, `optional`):
        dictionary of model metadata
    - PICKLE_FILE_EXTENSION (`str`):
        serialized model into `model.__name__.pkl`

    Usage example::

        TODO:
    """

    PICKLE_FILE_EXTENSION = ".pkl"

    def __init__(self, model: MT, metadata: t.Optional[t.Dict[str, t.Any]] = None):
        super(PickleArtifact, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType):
        f: Path = cls.get_path(path, cls.PICKLE_FILE_EXTENSION)
        with f.open("rb") as inf:
            model = cloudpickle.load(inf)
        return model

    def save(self, path: PathType) -> None:
        with open(self.model_path(path, self.PICKLE_FILE_EXTENSION), "wb") as inf:
            cloudpickle.dump(self._model, inf)
