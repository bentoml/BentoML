from .._internal.artifacts.base import MT, ModelArtifact
from .stores import _LocalStores


def save(name: str, artifact: ModelArtifact) -> str:
    ...


def load(name: str) -> MT:
    ...
