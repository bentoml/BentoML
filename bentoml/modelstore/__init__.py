from .._internal.artifacts.base import MT, ModelArtifact
from .manager import _ModelManager


def save(name: str, artifact: ModelArtifact) -> str:
    ...


def load(name: str) -> MT:
    ...
