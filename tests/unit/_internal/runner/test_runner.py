import pytest

import bentoml
from bentoml._internal.runner import Runner


class DummyRunnable(bentoml.Runnable):
    @bentoml.Runnable.method
    def dummy_runnable_method(self):
        pass


def test_runner():
    with pytest.raises(ValueError):
        Runner(DummyRunnable)

    named_runner = Runner(DummyRunnable, name="test_name")

    assert named_runner.name == "test_name"

    with pytest.raises(ValueError):
        Runner(DummyRunnable, name="invalid name")

    with pytest.raises(ValueError):
        Runner(DummyRunnable, name="invalid弁当name")

    with pytest.raises(ValueError):
        Runner(DummyRunnable, name="invalid!name")

    with pytest.raises(ValueError):
        Runner(DummyRunnable, name="invalid!name")
