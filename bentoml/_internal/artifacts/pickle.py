import os
import typing as t
from pathlib import Path

import cloudpickle

from ..exceptions import InvalidArgument
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
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
        pickle_module=cloudpickle,
    ):
        super(PickleArtifact, self).__init__(model, metadata=metadata)

        if isinstance(pickle_module, str):
            self._pickle = __import__(pickle_module)
        else:
            self._pickle = pickle_module

    def __pkl_path(self, base_path: PathType) -> PathType:
        return PathType(
            os.path.join(base_path, self._model.__str__() + PICKLE_FILE_EXTENSION)
        )

    @classmethod
    def load(cls, path: PathType) -> MT:
        pickle = object.__getattribute__(cls, "_pickle")
        model_path: Path = Path("")
        for f in Path(path).iterdir():
            if f.suffix == PICKLE_FILE_EXTENSION:
                model_path = f
                break
        if not model_path:
            raise InvalidArgument(f"Provided {path} doesn't contain pickle file.")
        with model_path.open("rb") as inf:
            model = pickle.load(inf)
        return model

    def save(self, path: PathType) -> None:
        with open(self.__pkl_path(path), "wb") as inf:
            self._pickle.dump(self._model, inf)
