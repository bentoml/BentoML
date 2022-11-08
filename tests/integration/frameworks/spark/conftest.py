import pytest

from bentoml._internal.bento.bento import Bento
from bentoml._internal.bento.build_config import BentoBuildConfig

from pyspark.sql.session import SparkSession

# create one spark session for all the tests in this module only
@pytest.fixture(scope="module")
def spark_session():
    return SparkSession.builder.master("local[1]").appName("testApp").getOrCreate()

# test with a simplebento
@pytest.fixture(scope="module")
def simplebento1() -> Bento:
    cfg = BentoBuildConfig("simplebento.py:svc")
    simplebento1 = Bento.create(cfg, build_ctx="./bento/simplebento")
    return simplebento1

# test with another copy of simplebento
@pytest.fixture(scope="module")
def simplebento2() -> Bento:
    cfg = BentoBuildConfig("simplebento.py:svc")
    simplebento2 = Bento.create(cfg, build_ctx="./bento/simplebento")
    return simplebento2
