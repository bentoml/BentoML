import os
import typing as t
from types import ModuleType

import cloudpickle

from ..types import MT, PathType
from .base import BaseArtifact

PICKLE_FILE_EXTENSION = ".pkl"


class PickleArtifact(BaseArtifact):
    """
    Abstraction for saving/loading python objects with pickle serialization
    """

    def __init__(
        self,
        model: MT,
        metadata=t.Optional[t.Dict[str, t.Any]],
        pickle_module=cloudpickle,
    ):
        super(PickleArtifact, self).__init__(model, metadata=metadata)

        if isinstance(pickle_module, str):
            self._pickle: ModuleType = __import__(pickle_module)
        else:
            self._pickle: ModuleType = pickle_module

        self._obj: MT = model

    def __pkl_path(self, base_path: PathType) -> PathType:
        return PathType(
            os.path.join(base_path, self._model.__str__ + PICKLE_FILE_EXTENSION)
        )

    @classmethod
    def load(cls, path: PathType) -> MT:
        with open(cls.__pkl_path(base_path=path), "rb") as inf:
            cls._obj = cls._pickle.load(inf)
        return cls._obj

    def save(self, dst: PathType) -> None:
        with open(self.__pkl_path(dst), "wb") as inf:
            self._pickle.dump(self._obj, inf)
