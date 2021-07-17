import os

import pytest

from bentoml._internal.artifacts import BaseArtifact

_metadata: t.Dict[str, t.Any] = {"test": "Hello", "num": 0.234}


class Foo(BaseArtifact):
    def __init__(self, model=None, metadata=None):
        super().__init__(model, metadata)
        if metadata is None:
            self._metadata = _metadata

    def save(self, path="/tmp"):
        return

    @classmethod
    def load(cls, path):
        model = object.__getattribute__(cls, '_model')
        return model


@pytest.mark.parametrize(
    "args, kwargs",
    [(["test"], {"metadata": _metadata}), ([Foo()], {"metadata": _metadata})],
)
def test_base_artifacts(args, kwargs):
    ba = BaseArtifact(*args, **kwargs)
    assert type(ba) == BaseArtifact
    assert ba.metadata == _metadata


@pytest.mark.parametrize("path, expected", [("/tmp")])
def test_save_artifacts():
    pass
