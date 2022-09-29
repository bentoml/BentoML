import typing as t
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.nodes import Item
    from _pytest.config import Config
    from _pytest.config.argparsing import Parser


def pytest_addoption(parser: "Parser") -> None:
    parser.addoption(
        "--disable-tf-eager-execution",
        action="store_true",
        default=False,
        help="Disable TF eager execution",
    )


def pytest_configure(config: "Config") -> None:
    # We will inject marker documentation here.
    config.addinivalue_line(
        "markers",
        "requires_eager_execution: requires enable eager execution to run Tensorflow-based tests.",
    )


def pytest_collection_modifyitems(config: "Config", items: t.List["Item"]) -> None:
    if config.getoption("--disable-tf-eager-execution"):
        try:
            from tensorflow.python.framework.ops import disable_eager_execution

            disable_eager_execution()
        except ImportError:
            return

    requires_eager_execution = pytest.mark.skip(reason="Requires eager execution")
    for item in items:
        if "requires_eager_execution" in item.keywords:
            item.add_marker(requires_eager_execution)
