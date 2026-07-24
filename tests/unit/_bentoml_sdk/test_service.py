from __future__ import annotations

import copy

import pytest

from _bentoml_sdk import service
from bentoml._internal.configuration.containers import BentoMLContainer


@pytest.fixture(autouse=True)
def reset_config():
    original_config = copy.deepcopy(BentoMLContainer.config.get())
    yield
    BentoMLContainer.config.set(original_config)


def test_inject_config_pollution_regression():
    @service(name="service_a", workers=2)
    class ServiceA:
        pass

    @service(name="service_b", workers=3)
    class ServiceB:
        pass

    ServiceA.inject_config()
    config_a_ref = BentoMLContainer.config.get()  # live reference
    workers_a_snapshot = config_a_ref["workers"]

    ServiceB.inject_config()

    assert config_a_ref["workers"] == workers_a_snapshot == 2
