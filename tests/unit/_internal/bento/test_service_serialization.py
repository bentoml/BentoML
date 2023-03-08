from __future__ import annotations

import os
import sys
from unittest.mock import ANY
from unittest.mock import MagicMock

import pytest
import cloudpickle

import bentoml
from bentoml.exceptions import NotFound
from bentoml._internal.tag import Tag
from bentoml._internal.configuration.containers import BentoMLContainer

from .test_bento import build_test_bento


@pytest.fixture
def build_bento():
    bento_store = BentoMLContainer.bento_store.get()
    tmp_store = BentoMLContainer.tmp_bento_store.get()
    working_dir = os.getcwd()
    bento = build_test_bento()
    os.chdir(working_dir)
    bento.save(bento_store)
    if "simplebento" in sys.modules:
        del sys.modules["simplebento"]
    yield bento
    try:
        bento_store.delete(bento.tag)
    except NotFound:
        pass
    try:
        tmp_store.delete(bento.tag)
    except NotFound:
        pass


@pytest.fixture
def reset_serialization_strategy():
    bentoml.set_serialization_strategy("EXPORT_BENTO")


@pytest.fixture
def reload_service_instance():
    if "simplebento" in sys.modules:
        del sys.modules["simplebento"]


@pytest.mark.usefixtures(
    "change_test_dir", "reset_serialization_strategy", "reload_service_instance"
)
def test_export_bento_strategy(build_bento):
    bentoml.set_serialization_strategy("EXPORT_BENTO")
    svc = bentoml.load(build_bento.tag)
    del sys.modules["simplebento"]
    loaded_svc = cloudpickle.loads(cloudpickle.dumps(svc))
    assert svc == loaded_svc


@pytest.mark.usefixtures(
    "change_test_dir", "reset_serialization_strategy", "reload_service_instance"
)
def test_local_bento_strategy(build_bento):
    bentoml.set_serialization_strategy("LOCAL_BENTO")
    svc = bentoml.load(build_bento.tag)
    del sys.modules["simplebento"]
    loaded_svc = cloudpickle.loads(cloudpickle.dumps(svc))
    assert svc == loaded_svc


@pytest.mark.usefixtures(
    "change_test_dir", "reset_serialization_strategy", "reload_service_instance"
)
def test_remote_bento_strategy_with_local_store_hit(build_bento):
    bentoml.set_serialization_strategy("REMOTE_BENTO")
    svc = bentoml.load(build_bento.tag)
    del sys.modules["simplebento"]
    loaded_svc = cloudpickle.loads(cloudpickle.dumps(svc))
    assert svc == loaded_svc


@pytest.mark.usefixtures(
    "change_test_dir", "reset_serialization_strategy", "reload_service_instance"
)
def test_remote_bento_strategy_with_tmp_store_hit(build_bento):
    bentoml.set_serialization_strategy("REMOTE_BENTO")
    svc = bentoml.load(build_bento.tag)
    serialized_svc = cloudpickle.dumps(svc)

    bento = bentoml.bentos.get(build_bento.tag)
    tmp_store = BentoMLContainer.tmp_bento_store.get()
    bento.save(bento_store=tmp_store)
    bentoml.bentos.delete(build_bento.tag)

    del sys.modules["simplebento"]
    assert svc == cloudpickle.loads(serialized_svc)


@pytest.mark.usefixtures(
    "change_test_dir", "reset_serialization_strategy", "reload_service_instance"
)
def test_remote_bento_strategy_pull_yatai(build_bento):
    bentoml.set_serialization_strategy("REMOTE_BENTO")
    svc = bentoml.load(build_bento.tag)

    mock_yatai_client = MagicMock()

    def pull_bento(*args, **kwargs):
        working_dir = os.getcwd()
        bento = build_test_bento()
        os.chdir(working_dir)
        bento.save()
        if "simplebento" in sys.modules:
            del sys.modules["simplebento"]

    mock_yatai_client.pull_bento.side_effect = pull_bento

    _yatai_client = BentoMLContainer.yatai_client.get()
    BentoMLContainer.yatai_client.set(mock_yatai_client)

    serialized_svc = cloudpickle.dumps(svc)
    bentoml.bentos.delete(build_bento.tag)

    del sys.modules["simplebento"]
    assert svc == cloudpickle.loads(serialized_svc)
    mock_yatai_client.pull_bento.assert_called_once_with(
        Tag("test.simplebento", "1.0"), bento_store=ANY
    )

    # Reset mock yatai_client
    BentoMLContainer.yatai_client.set(_yatai_client)
