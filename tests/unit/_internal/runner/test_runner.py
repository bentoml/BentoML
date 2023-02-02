import logging

import pytest

import bentoml
from bentoml._internal.runner import Runner


class DummyRunnable(bentoml.Runnable):
    @bentoml.Runnable.method
    def dummy_runnable_method(self):
        pass


def test_valid_runner_basic(caplog):
    dummy_runner = Runner(DummyRunnable)
    assert dummy_runner.name == "dummyrunnable"
    assert (
        "bentoml._internal.runner.runner",
        logging.WARNING,
        "Using lowercased runnable class name 'dummyrunnable' for runner.",
    ) in caplog.record_tuples

    named_runner = Runner(DummyRunnable, name="UPPERCASE_name")
    assert named_runner.name == "uppercase_name"
    assert (
        "bentoml._internal.runner.runner",
        logging.WARNING,
        "Converting runner name 'UPPERCASE_name' to lowercase: 'uppercase_name'",
    ) in caplog.record_tuples


def test_valid_runner_tag_with_dash():
    named_runner = Runner(DummyRunnable, name="test-name")
    assert named_runner.name == "test-name"


def test_valid_runner_tag_with_underscore():
    named_runner = Runner(DummyRunnable, name="test_name")
    assert named_runner.name == "test_name"


def test_valid_runner_tag_with_dot():
    named_runner = Runner(DummyRunnable, name="test.name")
    assert named_runner.name == "test.name"


def test_valid_runner_short_tag():
    named_runner = Runner(DummyRunnable, name="a")
    assert named_runner.name == "a"


def test_valid_runner_long_tag():
    named_runner = Runner(DummyRunnable, name="a" * 50)
    assert named_runner.name == "a" * 50

    # Boundary case: tag value must be at most 63 characters long
    named_runner = Runner(DummyRunnable, name="a" * 63)
    assert named_runner.name == "a" * 63


def test_valid_runner_tag_start_end_with_number():
    # Tag must begin and end with an alphanumeric character (`[a-z0-9A-Z]`)
    named_runner = Runner(DummyRunnable, name="9dummyrunnable")
    assert named_runner.name == "9dummyrunnable"

    named_runner = Runner(DummyRunnable, name="dummyrunnable9")
    assert named_runner.name == "dummyrunnable9"

    named_runner = Runner(DummyRunnable, name="9dummyrunnable9")
    assert named_runner.name == "9dummyrunnable9"


def test_invalid_runner_basic():
    with pytest.raises(ValueError):
        Runner(DummyRunnable, name="invalid name")

    with pytest.raises(ValueError):
        Runner(DummyRunnable, name="invalid弁当name")

    with pytest.raises(ValueError):
        Runner(DummyRunnable, name="invalid!name")


def test_invalid_runner_empty_tag():
    with pytest.raises(ValueError):
        Runner(DummyRunnable, name="")


def test_invalid_runner_long_tag():
    with pytest.raises(ValueError):
        Runner(DummyRunnable, name="a" * 100)

    # Boundary case: Tag value must be at most 63 characters long
    with pytest.raises(ValueError):
        Runner(DummyRunnable, name="a" * 64)


def test_invalid_runner_tag_not_start_end_with_alphanumeric():
    # Tag must begin and end with an alphanumeric character (`[a-z0-9A-Z]`)
    with pytest.raises(ValueError):
        Runner(DummyRunnable, name="_dummyrunnable")

    with pytest.raises(ValueError):
        Runner(DummyRunnable, name="dummyrunnable_")

    with pytest.raises(ValueError):
        Runner(DummyRunnable, name="_dummyrunnable_")
