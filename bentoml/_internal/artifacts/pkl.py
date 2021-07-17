import os
import typing as t
from pathlib import Path

from ..types import MT, PathType
from ..utils import cloudpickle
from .base import BaseArtifact

PICKLE_FILE_EXTENSION = ".pkl"


class PickleArtifact(BaseArtifact):
    """
    Abstraction for saving/loading python objects with pickle serialization.
    """

    def __init__(self, model: MT, metadata: t.Optional[t.Dict[str, t.Any]] = None):
        super(PickleArtifact, self).__init__(model, metadata=metadata)

    def _pkl_path(self, base_path: PathType) -> PathType:
        return PathType(
            os.path.join(base_path, self._model.__name__ + PICKLE_FILE_EXTENSION)
        )

    @classmethod
    def load(cls, path: PathType) -> MT:
        try:
            for f in Path(path).iterdir():
                if f.suffix == PICKLE_FILE_EXTENSION:
                    with f.open("rb") as inf:
                        model = cloudpickle.load(inf)
                    return model
        except FileNotFoundError:
            raise

    def save(self, path: PathType) -> None:
        with open(self._pkl_path(path), "wb") as inf:
            cloudpickle.dump(self._model, inf)
