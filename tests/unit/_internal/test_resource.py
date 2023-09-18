import math

import pytest

import bentoml
import bentoml.exceptions
from bentoml.exceptions import BentoMLConfigException
from bentoml._internal.resource import Resource
from bentoml._internal.resource import CpuResource
from bentoml._internal.resource import get_resource
from bentoml._internal.resource import system_resources
from bentoml._internal.resource import NvidiaGpuResource


class DummyResource(Resource[str], resource_id="dummy"):
    @classmethod
    def from_spec(cls, spec: str) -> str:
        return spec + "dummy"

    @classmethod
    def validate(cls, val: str):
        if val == "baddummy":
            raise ValueError

    @classmethod
    def from_system(cls) -> str:
        return "dummy_fromsystem"


def test_system_resources():
    assert system_resources()["dummy"] == "dummy_fromsystem"


def test_get_resource():
    with pytest.raises(bentoml.exceptions.BentoMLConfigException):
        get_resource({}, "bad_resource")

    with pytest.raises(ValueError):
        get_resource({"dummy": "bad"}, "dummy")

    assert get_resource({}, "dummy") is None
    assert get_resource({"dummy": "testval"}, "dummy") == "testvaldummy"
    assert get_resource({"dummy": "system"}, "dummy") == "dummy_fromsystem"


def test_CpuResource():
    assert CpuResource.from_system() > 0  # TODO: real from_system tests
    with pytest.raises(BentoMLConfigException):
        CpuResource.validate(CpuResource.from_system() + 1)
    with pytest.raises(BentoMLConfigException):
        CpuResource.validate(-1)

    CpuResource.validate(0)
    CpuResource.validate(0.1)
    CpuResource.validate(1)

    assert math.isclose(CpuResource.from_spec("100m"), 0.1)
    assert math.isclose(CpuResource.from_spec(0.3), 0.3)
    assert math.isclose(CpuResource.from_spec(1), 1.0)
    assert math.isclose(CpuResource.from_spec("5"), 5.0)

    with pytest.raises(TypeError):
        CpuResource.from_spec((1, 2, 3))


def test_NvidiaGpuResource():
    assert len(NvidiaGpuResource.from_system()) >= 0  # TODO: real from_system tests

    with pytest.raises(BentoMLConfigException):
        NvidiaGpuResource.validate(NvidiaGpuResource.from_system() + [1])
    with pytest.raises(BentoMLConfigException):
        NvidiaGpuResource.validate([-2])
    with pytest.raises(BentoMLConfigException):
        NvidiaGpuResource.validate([-1])

    NvidiaGpuResource.validate([])
    # NvidiaGpuResource.validate(1)  # TODO: work out how to skip this on systems with no GPU

    assert NvidiaGpuResource.from_spec(1) == [0]
    assert NvidiaGpuResource.from_spec("5") == [0, 1, 2, 3, 4]
    assert NvidiaGpuResource.from_spec(1) == [0]
    assert NvidiaGpuResource.from_spec(2) == [0, 1]
    assert NvidiaGpuResource.from_spec("3") == [0, 1, 2]
    assert NvidiaGpuResource.from_spec([1, 3]) == [1, 3]
    assert NvidiaGpuResource.from_spec(["1", "3"]) == [1, 3]
    assert NvidiaGpuResource.from_spec(-1) == []
    assert NvidiaGpuResource.from_spec("-1") == []

    with pytest.raises(BentoMLConfigException):
        # Currently this is not supported and is considered invalid
        assert NvidiaGpuResource.from_spec("[1, 2, 3]") == [1, 2, 3]

    with pytest.raises(TypeError):
        NvidiaGpuResource.from_spec((1, 2, 3))

    with pytest.raises(BentoMLConfigException):
        NvidiaGpuResource.from_spec("100m")

    with pytest.raises(BentoMLConfigException):
        assert NvidiaGpuResource.from_spec(-2)

    with pytest.raises(BentoMLConfigException):
        assert NvidiaGpuResource.from_spec("-2")

    with pytest.raises(TypeError):
        NvidiaGpuResource.from_spec(1.5)
