# pylint: disable=redefined-outer-name
# type: ignore[no-untyped-def]

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

import bentoml
from bentoml.io import Text
from bentoml.exceptions import BentoMLException

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


svc = bentoml.Service(name="general_features_routes")


@svc.api(input=Text(), output=Text(), route="api/v1/test_without_prefix")
async def without_prefix(obj: str) -> int:
    return 1


@svc.api(input=Text(), output=Text(), route="/api/v1/with_prefix")
async def with_prefix(obj: str) -> int:
    return 1


def test_routes_prefix():
    assert all(i.path.startswith("/") for i in svc.asgi_app.routes)


def test_invalid_api_naming():
    svc = bentoml.Service(name="invalid_api_naming")

    @svc.api(name="similar", input=Text(), output=Text())
    def r1(obj: str) -> int:
        return 1

    with pytest.raises(BentoMLException) as excinfo:

        @svc.api(name="similar", input=Text(), output=Text())
        def r2(obj: str) -> int:
            return 1

    assert "is already defined in Service" in str(excinfo.value)


def test_get_valid_service_name(caplog: LogCaptureFixture):
    from bentoml._internal.service.service import get_valid_service_name

    with caplog.at_level(logging.WARNING):
        get_valid_service_name("NOT_LOWER")
    assert "to lowercase:" in caplog.text


def test_service_init_checks():
    py_model1 = bentoml.picklable_model.get("py_model.case-1.e2e").to_runner(
        name="invalid"
    )
    py_model2 = bentoml.picklable_model.get("py_model.case-1.e2e").to_runner(
        name="invalid"
    )
    with pytest.raises(ValueError) as excinfo:
        svc = bentoml.Service(name="duplicates_runners", runners=[py_model1, py_model2])
    assert "Found duplicate name" in str(excinfo.value)

    with pytest.raises(AssertionError) as excinfo:
        svc = bentoml.Service(name="invalid_model_type", models=[1])
    assert "Service models list can only" in str(excinfo.value)


def test_dunder_string():
    runner = bentoml.picklable_model.get("py_model.case-1.e2e").to_runner()

    svc = bentoml.Service(name="dunder_string", runners=[runner])

    assert (
        str(svc)
        == 'bentoml.Service(name="dunder_string", runners=[py_model.case-1.e2e])'
    )
