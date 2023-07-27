from __future__ import annotations

import sys
import typing as t
from unittest.mock import ANY
from unittest.mock import MagicMock

import cloudpickle
import pytest

import bentoml
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.tag import Tag
from bentoml.exceptions import NotFound

from .test_bento import build_test_bento


@pytest.fixture
def build_bento() -> t.Generator[bentoml.Bento, None, None]:
    bento_store = BentoMLContainer.bento_store.get()
    model_store = BentoMLContainer.model_store.get()
    tmp_store = BentoMLContainer.tmp_bento_store.get()
    bento = build_test_bento()
    try:
        bento_store.get(bento.tag)
    except NotFound:
        bento.save(bento_store, model_store)
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
def test_export_bento_strategy(build_bento: bentoml.Bento):
    bentoml.set_serialization_strategy("EXPORT_BENTO")
    svc = bentoml.load(build_bento.tag)
    del sys.modules["simplebento"]
    loaded_svc = cloudpickle.loads(cloudpickle.dumps(svc))
    assert svc == loaded_svc


@pytest.mark.usefixtures(
    "change_test_dir", "reset_serialization_strategy", "reload_service_instance"
)
def test_local_bento_strategy(build_bento: bentoml.Bento):
    bentoml.set_serialization_strategy("LOCAL_BENTO")
    svc = bentoml.load(build_bento.tag)
    del sys.modules["simplebento"]
    loaded_svc = cloudpickle.loads(cloudpickle.dumps(svc))
    assert svc == loaded_svc


@pytest.mark.usefixtures(
    "change_test_dir", "reset_serialization_strategy", "reload_service_instance"
)
def test_remote_bento_strategy_with_local_store_hit(build_bento: bentoml.Bento):
    bentoml.set_serialization_strategy("REMOTE_BENTO")
    svc = bentoml.load(build_bento.tag)
    del sys.modules["simplebento"]
    loaded_svc = cloudpickle.loads(cloudpickle.dumps(svc))
    assert svc == loaded_svc


@pytest.mark.usefixtures(
    "change_test_dir", "reset_serialization_strategy", "reload_service_instance"
)
def test_remote_bento_strategy_with_tmp_store_hit(build_bento: bentoml.Bento):
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
def test_remote_bento_strategy_pull_yatai(build_bento: bentoml.Bento):
    bentoml.set_serialization_strategy("REMOTE_BENTO")
    svc = bentoml.load(build_bento.tag)

    mock_bentocloud_client = MagicMock()

    def pull_bento(*args: t.Any, **kwargs: t.Any):
        bento = build_test_bento()
        if str(bento.tag) not in [str(b.tag) for b in bentoml.bentos.list()]:
            bento.save()
        if "simplebento" in sys.modules:
            del sys.modules["simplebento"]

    mock_bentocloud_client.pull_bento.side_effect = pull_bento

    _bentocloud_client = BentoMLContainer.bentocloud_client.get()
    BentoMLContainer.bentocloud_client.set(mock_bentocloud_client)

    serialized_svc = cloudpickle.dumps(svc)
    bentoml.bentos.delete(build_bento.tag)

    del sys.modules["simplebento"]
    assert svc == cloudpickle.loads(serialized_svc)
    mock_bentocloud_client.pull_bento.assert_called_once_with(
        Tag("test.simplebento", "1.0"), bento_store=ANY
    )

    # Reset mock bentocloud_client
    BentoMLContainer.bentocloud_client.set(_bentocloud_client)
