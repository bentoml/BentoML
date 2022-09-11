from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

import pytest

from bentoml._internal.models import ModelStore

if TYPE_CHECKING:
    from _pytest.main import Session
    from _pytest.nodes import Item
    from _pytest.config import Config
    from _pytest.config.argparsing import Parser


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--gpus", action="store_true", default=False, help="run gpus related tests"
    )
    parser.addoption(
        "--disable-tf-eager-execution",
        action="store_true",
        default=False,
        help="Disable TF eager execution",
    )


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    if config.getoption("--disable-tf-eager-execution"):
        try:
            from tensorflow.python.framework.ops import disable_eager_execution

            disable_eager_execution()
        except ImportError:
            return
    elif config.getoption("--gpus"):
        return

    skip_gpus = pytest.mark.skip(reason="Skip gpus tests")
    requires_eager_execution = pytest.mark.skip(reason="Requires eager execution")
    for item in items:
        if "gpus" in item.keywords:
            item.add_marker(skip_gpus)
        if "requires_eager_execution" in item.keywords:
            item.add_marker(requires_eager_execution)


def pytest_sessionstart(session: Session):  # pylint: disable=unused-argument
    path = tempfile.mkdtemp("bentoml-pytest-unit")
    from bentoml._internal.configuration.containers import BentoMLContainer

    BentoMLContainer.model_store.set(ModelStore(path))
