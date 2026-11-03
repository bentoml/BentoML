import math

import pytest

import bentoml
import bentoml.exceptions
from bentoml._internal.resource import CpuResource
from bentoml._internal.resource import NvidiaGpuResource
from bentoml._internal.resource import Resource
from bentoml._internal.resource import get_resource
from bentoml._internal.resource import system_resources
from bentoml.exceptions import BentoMLConfigException


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


def test_query_cgroup_cpu_count_respects_affinity(monkeypatch):
    # A process pinned to a subset of the host's cores (taskset,
    # docker --cpuset-cpus, a cpuset) must not be reported the host count.
    from bentoml._internal import resource

    resource.query_cgroup_cpu_count.cache_clear()
    monkeypatch.setattr(
        resource.os, "sched_getaffinity", lambda pid: {0, 1}, raising=False
    )
    # No cgroup quota, so only the available-CPU count caps the result.
    monkeypatch.setattr(resource.os.path, "exists", lambda p: False)
    monkeypatch.setattr(resource.os, "cpu_count", lambda: 64)
    try:
        assert resource.query_cgroup_cpu_count() == 2.0
    finally:
        resource.query_cgroup_cpu_count.cache_clear()


def test_query_cgroup_cpu_count_falls_back_without_affinity(monkeypatch):
    # Where the affinity mask is unavailable (non-Linux), fall back to the
    # host CPU count, preserving the previous behaviour.
    from bentoml._internal import resource

    resource.query_cgroup_cpu_count.cache_clear()
    monkeypatch.delattr(resource.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(resource.os.path, "exists", lambda p: False)
    monkeypatch.setattr(resource.os, "cpu_count", lambda: 8)
    try:
        assert resource.query_cgroup_cpu_count() == 8.0
    finally:
        resource.query_cgroup_cpu_count.cache_clear()


def test_query_cgroup_cpu_count_quota_caps_affinity(monkeypatch):
    # A cgroup v2 CPU quota (docker --cpus) still caps below the affinity count.
    import io

    from bentoml._internal import resource

    resource.query_cgroup_cpu_count.cache_clear()
    monkeypatch.setattr(
        resource.os, "sched_getaffinity", lambda pid: set(range(8)), raising=False
    )
    monkeypatch.setattr(resource.os.path, "exists", lambda p: p.endswith("cpu.max"))
    real_open = open
    monkeypatch.setattr(
        "builtins.open",
        lambda p, *a, **k: (
            io.StringIO("200000 100000\n")
            if str(p).endswith("cpu.max")
            else real_open(p, *a, **k)
        ),
    )
    monkeypatch.setattr(resource.os, "cpu_count", lambda: 64)
    try:
        assert resource.query_cgroup_cpu_count() == 2.0
    finally:
        resource.query_cgroup_cpu_count.cache_clear()
