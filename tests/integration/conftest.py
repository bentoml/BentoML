import pytest

from bentoml import ModelStore


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--gpus", action="store_true", default=False, help="run gpus related tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--gpus"):
        return
    skip_gpus = pytest.mark.skip(reason="need --gpus option to run")
    for item in items:
        if "gpus" in item.keywords:
            item.add_marker(skip_gpus)


@pytest.fixture(scope="session")
def modelstore(tmp_path_factory) -> "ModelStore":
    # we need to get consistent cache folder, thus tmpdir is not usable here
    # NOTE: after using modelstore, also use `delete_cache_model` to remove model after
    #  load tests.
    path = tmp_path_factory.mktemp("bentoml")
    return ModelStore(base_dir=path)
