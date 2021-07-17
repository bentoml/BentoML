import os

import pytest

from bentoml._internal.artifacts import BaseArtifact, PickleArtifact
from bentoml._internal.exceptions import InvalidArgument

_metadata = {"test": "Hello", "num": 0.234}


def create_mock_class(name):
    class Foo:
        n = 1

    Foo.__name__ = name
    return Foo


class FooArtifact(BaseArtifact):
    def __init__(self, model, metadata=None):
        super().__init__(model, metadata)
        if metadata is None:
            self._metadata = _metadata

    def save(self, path):
        return

    @classmethod
    def load(cls, path):
        return "foo"


@pytest.mark.parametrize(
    "args, kwargs, metadata",
    [
        ([create_mock_class("foo")], {"metadata": _metadata}, _metadata),
        ([create_mock_class("bar")], {}, None),
        ([b'\x00'], {}, None),
        (['test'], {}, None),
        ([1], {}, None),
    ],
)
def test_base_artifact(args, kwargs, metadata):
    ba = BaseArtifact(*args, **kwargs)
    pkl = PickleArtifact(*args, **kwargs)
    assert all(
        [v == k] for k in [ba.__dict__, pkl.__dict__] for v in ['_model', '_metadata']
    )
    assert ba.metadata == metadata
    assert pkl.metadata == metadata


@pytest.mark.parametrize(
    "model, expected",
    [
        (create_mock_class("MockModel"), "MockModel.yml"),
        (create_mock_class("test"), "test.yml"),
        (create_mock_class("1"), "1.yml"),
    ],
)
def test_save_artifact(model, expected, tmpdir):
    foo = FooArtifact(model, metadata=_metadata)
    foo.save(tmpdir)
    assert foo.model_path(tmpdir, ".yml") == os.path.join(tmpdir, expected)
    assert os.path.exists(foo.model_path(tmpdir, ".yml"))


@pytest.mark.parametrize(
    "model, expected",
    [
        (create_mock_class("MockModel"), "MockModel.pkl"),
        (create_mock_class("test"), "test.pkl"),
    ],
)
def test_pkl_artifact(model, expected, tmpdir):
    pkl = PickleArtifact(model, metadata=_metadata)
    pkl.save(tmpdir)
    assert pkl.model_path(tmpdir, ".pkl") == os.path.join(tmpdir, expected)
    assert os.path.exists(pkl.model_path(tmpdir, ".pkl"))
    assert os.path.exists(pkl.model_path(tmpdir, ".yml"))
    assert model == PickleArtifact.load(tmpdir)


@pytest.mark.parametrize(
    "path, exc", [("test", FileNotFoundError), ("/tmp/test", FileNotFoundError)]
)
def test_empty_pkl_path(path, exc):
    with pytest.raises(exc):
        PickleArtifact.load(path)
