import typing as t
from typing import TYPE_CHECKING

import pytest

from bentoml._internal.models import ModelStore

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.config.argparsing import Parser
    from _pytest.nodes import Item
    from _pytest.tmpdir import TempPathFactory


def pytest_addoption(parser: "Parser") -> None:
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--gpus", action="store_true", default=False, help="run gpus related tests"
    )


def pytest_collection_modifyitems(config: "Config", items: t.List["Item"]) -> None:
    if config.getoption("--gpus"):
        return
    skip_gpus = pytest.mark.skip(reason="need --gpus option to run")
    for item in items:
        if "gpus" in item.keywords:
            item.add_marker(skip_gpus)


@pytest.fixture(scope="session", name="modelstore")
def fixture_modelstore(tmp_path_factory: "TempPathFactory") -> "ModelStore":
    # we need to get consistent cache folder, thus tmpdir is not usable here
    # NOTE: after using modelstore, also use `delete_cache_model` to remove model after
    #  load tests.
    path = tmp_path_factory.mktemp("bentoml")
    return ModelStore(path)
