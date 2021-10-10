# pylint: disable=redefined-outer-name
import contextlib
import os

import pytest

from bentoml import ModelStore

bentoml_cache_dir = os.getenv(
    "BENTOML_CACHE_DIR",
    os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "bentoml"),
)


@pytest.fixture(scope="session")
def clean_context():
    with contextlib.ExitStack() as stack:
        yield stack


@pytest.fixture(scope="session")
def modelstore() -> "ModelStore":
    # we need to get consistent cache folder, thus tmpdir is not usable here
    # NOTE: after using modelstore, also use `delete_cache_model` to remove model after
    #  load tests.
    return ModelStore(base_dir=bentoml_cache_dir)


@pytest.fixture(scope="session")
def cleanup():
    yield None
    os.system(f"rm -rf {bentoml_cache_dir}/*")
