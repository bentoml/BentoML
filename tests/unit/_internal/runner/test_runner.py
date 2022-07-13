import logging

import pytest

import bentoml
from bentoml._internal.runner import Runner


class DummyRunnable(bentoml.Runnable):
    @bentoml.Runnable.method
    def dummy_runnable_method(self):
        pass


def test_runner(caplog):
    dummy_runner = Runner(DummyRunnable)

    assert dummy_runner.name == "dummyrunnable"
    assert (
        "bentoml._internal.runner.runner",
        logging.WARNING,
        "Using lowercased runnable class name 'dummyrunnable' for runner.",
    ) in caplog.record_tuples

    named_runner = Runner(DummyRunnable, name="UPPERCASE_name")
    assert (
        "bentoml._internal.runner.runner",
        logging.WARNING,
        "Converting runner name 'UPPERCASE_name' to lowercase: 'uppercase_name'",
    ) in caplog.record_tuples

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
