import typing as t
from ..types import MT as MT, PathType as PathType
from ..utils import cloudpickle as cloudpickle
from .base import BaseArtifact as BaseArtifact

PICKLE_FILE_EXTENSION: str

class PickleArtifact(BaseArtifact):
    def __init__(self, model: MT, metadata: t.Optional[t.Dict[str, t.Any]] = ...) -> None: ...
    @classmethod
    def load(cls, path: PathType): ...
    def save(self, path: PathType) -> None: ...
