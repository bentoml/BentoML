import os

import pytest

from bentoml._internal.artifacts import ModelArtifact, PickleArtifact

_metadata = {"test": "Hello", "num": 0.234}


def create_mock_class(name):
    class Foo:
        n = 1

    if name:
        Foo.__name__ = name
    return Foo


class FooArtifact(ModelArtifact):
    def __init__(self, model, metadata=None):
        super().__init__(model, metadata)
        if metadata is None:
            self._metadata = _metadata

    def save(self, path):
        return self.get_path(path, ".est")

    @classmethod
    def load(cls, path):
        return 'foo'


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
    ba = ModelArtifact(*args, **kwargs)
    pkl = PickleArtifact(*args, **kwargs)
    assert all(
        [v == k] for k in [ba.__dict__, pkl.__dict__] for v in ['_model', '_metadata']
    )
    assert ba.metadata == metadata
    assert pkl.metadata == metadata


@pytest.mark.parametrize(
    "model, expected",
    [
        (create_mock_class("MockModel"), "model.yml"),
        (create_mock_class("test"), "model.yml"),
        (create_mock_class("1"), "model.yml"),
    ],
)
def test_save_artifact(model, expected, tmpdir):
    foo = FooArtifact(model, metadata=_metadata)
    foo.save(tmpdir)
    assert os.path.exists(foo.get_path(tmpdir, ".yml"))


@pytest.mark.parametrize(
    "model, expected",
    [
        (create_mock_class("MockModel"), "model.pkl"),
        (create_mock_class("test"), "model.pkl"),
    ],
)
def test_pkl_artifact(model, expected, tmpdir):
    pkl = PickleArtifact(model, metadata=_metadata)
    pkl.save(tmpdir)
    assert os.path.exists(pkl.get_path(tmpdir, ".pkl"))
    assert os.path.exists(pkl.get_path(tmpdir, ".yml"))
    assert model == PickleArtifact.load(tmpdir)
