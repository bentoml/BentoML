import typing as t
import tempfile
from typing import TYPE_CHECKING

import pytest

from bentoml._internal.models import ModelStore

if TYPE_CHECKING:
    from _pytest.nodes import Item
    from _pytest.config import Config
    from _pytest.config.argparsing import Parser


def pytest_addoption(parser: "Parser") -> None:
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--gpus", action="store_true", default=False, help="run gpus related tests"
    )
    parser.addoption(
        "--disable-eager-execution",
        action="store_true",
        default=False,
        help="Disable TF eager execution",
    )


def pytest_collection_modifyitems(config: "Config", items: t.List["Item"]) -> None:
    if config.getoption("--disable-eager-execution"):
        from tensorflow.python.framework.ops import disable_eager_execution

        disable_eager_execution()
    elif config.getoption("--gpus"):
        return

    skip_gpus = pytest.mark.skip(reason="need --gpus option to run")
    for item in items:
        if "gpus" in item.keywords:
            item.add_marker(skip_gpus)


def pytest_sessionstart(session):
    path = tempfile.mkdtemp("bentoml-pytest")
    from bentoml._internal.configuration.containers import BentoMLContainer

    BentoMLContainer.model_store.set(ModelStore(path))
