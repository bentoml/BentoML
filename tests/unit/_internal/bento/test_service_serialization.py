from __future__ import annotations


import pytest
import os
import bentoml
import cloudpickle

from .test_bento import build_test_bento
from bentoml.exceptions import NotFound
from bentoml._internal.configuration.containers import BentoMLContainer

@pytest.fixture
def build_bento():
    working_dir = os.getcwd()
    bento = build_test_bento()
    os.chdir(working_dir)
    bento.save()
    yield bento
    try:
        bentoml.bentos.delete(bento.tag)
    except NotFound:
        pass
    try:
        tmp_store = BentoMLContainer.tmp_bento_store.get()
        tmp_store.delete(bento.tag)
    except NotFound:
        pass

@pytest.fixture
def reset_serialization_strategy():
    bentoml.set_serialization_strategy("EXPORT_BENTO")

@pytest.mark.usefixtures("change_test_dir", "reset_serialization_strategy")
def test_export_bento_strategy(build_bento):
    bentoml.set_serialization_strategy("EXPORT_BENTO")
    svc = bentoml.load(build_bento.tag)
    loaded_svc = cloudpickle.loads(cloudpickle.dumps(svc))
    assert svc == loaded_svc

@pytest.mark.usefixtures("change_test_dir", "reset_serialization_strategy")
def test_local_bento_strategy(build_bento):
    bentoml.set_serialization_strategy("LOCAL_BENTO")
    svc = bentoml.load(build_bento.tag)
    loaded_svc = cloudpickle.loads(cloudpickle.dumps(svc))
    assert svc == loaded_svc

