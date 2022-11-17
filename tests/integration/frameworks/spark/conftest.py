import pytest
from pyspark.sql.session import SparkSession

import bentoml
from bentoml._internal.bento.bento import Bento
from bentoml._internal.bento.bento import BentoStore


# create one spark session for all the tests
@pytest.fixture(scope="session")
def spark():
    return (
        SparkSession.builder.config("spark.files.overwrite", "true")
        .master("local[1]")
        .appName("testApp")
        .getOrCreate()
    )


# test with a temporary bento store
@pytest.fixture(
    scope="function"
)  # setting scope to 'function' because tmpdir is a function-scoped fixture
def bento_store(tmpdir: str) -> BentoStore:
    return BentoStore(tmpdir)


# test with a simplebento
@pytest.fixture(scope="function")
def simplebento(bento_store: BentoStore) -> Bento:
    return bentoml.bentos.build(
        "simplebento.py:svc",
        build_ctx="./tests/unit/_internal/bento/simplebento",
        _bento_store=bento_store,
    )
