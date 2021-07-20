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
        return self.walk_path(path, ".est")

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
    assert os.path.exists(foo.model_path(tmpdir, ".yml"))


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
    assert os.path.exists(pkl.model_path(tmpdir, ".pkl"))
    assert os.path.exists(pkl.model_path(tmpdir, ".yml"))
    assert model == PickleArtifact.load(tmpdir)


@pytest.mark.parametrize(
    "path, exc", [("test", FileNotFoundError), ("/tmp/test", FileNotFoundError)]
)
def test_empty_walk_path(path, exc):
    with pytest.raises(exc):
        FooArtifact("1").save(path)


def test_valid_walk_path(tmpdir):
    os.mknod(os.path.join(tmpdir, "t.est"))
    assert os.path.exists(FooArtifact('1').save(tmpdir))


def test_dir_walk_path(tmpdir):
    p = os.path.join(tmpdir, "test", "foo")
    os.makedirs(p, exist_ok=True)
    os.mknod(os.path.join(p, "t.est"))
    assert os.path.exists(FooArtifact('1').save(p))
