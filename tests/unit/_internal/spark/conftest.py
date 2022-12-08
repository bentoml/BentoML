import pytest

import bentoml
from bentoml._internal.bento.bento import Bento
from bentoml._internal.bento.bento import BentoStore


@pytest.fixture(scope="function")
def bento_store(tmpdir: str) -> BentoStore:
    return BentoStore(tmpdir)


@pytest.fixture(scope="function")
def test_bento(bento_store: BentoStore) -> Bento:
    return bentoml.Bento.import_from("tests/unit/_internal/spark/test_bento.bento")
