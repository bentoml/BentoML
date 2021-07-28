import os

import pytest

from bentoml._internal.models import Model, PickleModel
from tests._internal.helpers import assert_have_file_extension

_metadata = {"test": "Hello", "num": 0.234}


def create_mock_class(name):
    class Foo:
        n = 1

    if name:
        Foo.__name__ = name
    return Foo


class FooModel(Model):
    def __init__(self, model, metadata=None):
        super().__init__(model, metadata)
        if metadata is None:
            self._metadata = _metadata

    def save(self, path):
        return os.path.join(path, "model.test")

    @classmethod
    def load(cls, path):
        return "foo"


class InvalidModel(Model):
    """InvalidModel doesn't have save and load implemented"""

    def __init__(self, model=None):
        super().__init__(model)


@pytest.mark.parametrize(
    "args, kwargs, metadata",
    [
        ([create_mock_class("foo")], {"metadata": _metadata}, _metadata),
        ([create_mock_class("bar")], {}, None),
        ([b"\x00"], {}, None),
        (["test"], {}, None),
        ([1], {}, None),
    ],
)
def test_base_artifact(args, kwargs, metadata):
    ba = Model(*args, **kwargs)
    pkl = PickleModel(*args, **kwargs)
    assert all(
        [v == k] for k in [ba.__dict__, pkl.__dict__] for v in ["_model", "_metadata"]
    )
    assert ba.metadata == metadata
    assert pkl.metadata == metadata


@pytest.mark.parametrize(
    "model",
    [
        (create_mock_class("MockModel")),
        (create_mock_class("test")),
        (create_mock_class("1")),
    ],
)
def test_save_artifact(model, tmpdir):
    foo = FooModel(model, metadata=_metadata)
    foo.save(tmpdir)
    assert_have_file_extension(tmpdir, ".yml")


@pytest.mark.parametrize(
    "model", [(create_mock_class("MockModel")), (create_mock_class("test"))],
)
def test_pkl_artifact(model, tmpdir):
    pkl = PickleModel(model, metadata=_metadata)
    pkl.save(tmpdir)
    assert model == PickleModel.load(tmpdir)
    assert_have_file_extension(tmpdir, ".pkl")
    assert_have_file_extension(tmpdir, ".yml")


@pytest.mark.parametrize(
    "func, exc",
    [
        (InvalidModel().save, NotImplementedError),
        (InvalidModel.load, NotImplementedError),
    ],
)
def test_invalid_impl(func, exc):
    with pytest.raises(exc):
        func("/tmp/test")
