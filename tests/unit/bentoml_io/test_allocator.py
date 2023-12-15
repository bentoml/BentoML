from unittest import mock

import pytest

from _bentoml_impl.server.allocator import BentoMLConfigException
from _bentoml_impl.server.allocator import ResourceAllocator
from _bentoml_impl.server.allocator import ResourceUnavailable
from _bentoml_sdk import service


@mock.patch(
    "_bentoml_impl.server.allocator.system_resources",
    return_value={"cpu": 8, "nvidia.com/gpu": list(range(4))},
)
def test_assign_gpus(_):
    s = ResourceAllocator()

    result = s.assign_gpus(2)
    assert result == [0, 1]

    result = s.assign_gpus(1)
    assert result == [2]

    result = s.assign_gpus(1)
    assert result == [3]

    with pytest.raises(ResourceUnavailable):
        s.assign_gpus(1)


@mock.patch(
    "_bentoml_impl.server.allocator.system_resources",
    return_value={"cpu": 8, "nvidia.com/gpu": list(range(4))},
)
def test_assign_gpus_float(_):
    s = ResourceAllocator()

    assert s.assign_gpus(0.5) == [0]
    assert s.assign_gpus(1) == [1]
    assert s.assign_gpus(0.5) == [0]
    assert s.assign_gpus(0.5) == [2]
    for _ in range(4):
        assert s.assign_gpus(0.25) == [3]

    with pytest.raises(ResourceUnavailable):
        s.assign_gpus(1)
    with pytest.raises(BentoMLConfigException):
        s.assign_gpus(1.5)


@mock.patch(
    "_bentoml_impl.server.allocator.system_resources",
    return_value={"cpu": 8, "nvidia.com/gpu": list(range(4))},
)
def test_get_worker_env_gpu(_):
    s = ResourceAllocator()

    @service(resources={"gpu": 2})
    class Foo:
        pass

    Foo.inject_config()
    num_workers, worker_env = s.get_worker_env(Foo)
    assert num_workers == 2
    assert worker_env[0]["CUDA_VISIBLE_DEVICES"] == "0"
    assert worker_env[1]["CUDA_VISIBLE_DEVICES"] == "1"

    num_workers, worker_env = s.get_worker_env(Foo)
    assert num_workers == 2
    assert worker_env[0]["CUDA_VISIBLE_DEVICES"] == "2"
    assert worker_env[1]["CUDA_VISIBLE_DEVICES"] == "3"


@mock.patch(
    "_bentoml_impl.server.allocator.system_resources",
    return_value={"cpu": 8, "nvidia.com/gpu": list(range(4))},
)
def test_get_worker_env_gpu_float(_):
    s = ResourceAllocator()

    @service(resources={"gpu": 0.5})
    class Foo:
        pass

    @service(resources={"gpu": 2})
    class Bar:
        pass

    Foo.inject_config()
    num_workers, worker_env = s.get_worker_env(Foo)
    assert num_workers == 1
    assert worker_env[0]["CUDA_VISIBLE_DEVICES"] == "0"

    num_workers, worker_env = s.get_worker_env(Foo)
    assert num_workers == 1
    assert worker_env[0]["CUDA_VISIBLE_DEVICES"] == "0"
    Bar.inject_config()
    num_workers, worker_env = s.get_worker_env(Bar)
    assert num_workers == 2
    assert worker_env[0]["CUDA_VISIBLE_DEVICES"] == "1"
    assert worker_env[1]["CUDA_VISIBLE_DEVICES"] == "2"


@mock.patch(
    "_bentoml_impl.server.allocator.system_resources",
    return_value={"cpu": 8, "nvidia.com/gpu": list(range(4))},
)
def test_get_worker_env_cpu_count(_):
    s = ResourceAllocator()

    @service(workers="cpu_count")
    class Foo:
        pass

    Foo.inject_config()
    num_workers, worker_env = s.get_worker_env(Foo)
    assert num_workers == 8
    assert not worker_env


@mock.patch(
    "_bentoml_impl.server.allocator.system_resources",
    return_value={"cpu": 8, "nvidia.com/gpu": list(range(4))},
)
def test_get_worker_env_worker_number(_):
    s = ResourceAllocator()

    @service(resources={"gpu": 2}, workers=2)
    class Foo:
        pass

    @service(resources={"gpu": 0.5}, workers=2)
    class Bar:
        pass

    Foo.inject_config()
    num_workers, worker_env = s.get_worker_env(Foo)
    assert num_workers == 2
    assert worker_env[0]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert worker_env[1]["CUDA_VISIBLE_DEVICES"] == "0,1"
    Bar.inject_config()
    num_workers, worker_env = s.get_worker_env(Bar)
    assert num_workers == 2
    assert worker_env[0]["CUDA_VISIBLE_DEVICES"] == "2"
    assert worker_env[1]["CUDA_VISIBLE_DEVICES"] == "2"

    num_workers, worker_env = s.get_worker_env(Bar)
    assert num_workers == 2
    assert worker_env[0]["CUDA_VISIBLE_DEVICES"] == "2"
    assert worker_env[1]["CUDA_VISIBLE_DEVICES"] == "2"


@mock.patch(
    "_bentoml_impl.server.allocator.system_resources",
    return_value={"cpu": 8, "nvidia.com/gpu": list(range(4))},
)
def test_get_worker_env_worker_gpu(_):
    s = ResourceAllocator()

    @service(workers=[{"gpus": 2}, {"gpus": 1}])
    class Foo:
        pass

    @service(workers=[{"gpus": 0.5}, {"gpus": 0.5}])
    class Bar:
        pass

    Foo.inject_config()
    num_workers, worker_env = s.get_worker_env(Foo)
    assert num_workers == 2
    assert worker_env[0]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert worker_env[1]["CUDA_VISIBLE_DEVICES"] == "2"

    with pytest.raises(ResourceUnavailable):
        s.get_worker_env(Foo)
    Bar.inject_config()
    num_workers, worker_env = s.get_worker_env(Bar)
    assert num_workers == 2
    assert worker_env[0]["CUDA_VISIBLE_DEVICES"] == "3"
    assert worker_env[1]["CUDA_VISIBLE_DEVICES"] == "3"


@mock.patch(
    "_bentoml_impl.server.allocator.system_resources",
    return_value={"cpu": 8, "nvidia.com/gpu": list(range(4))},
)
def test_get_worker_env_gpu_id(_):
    s = ResourceAllocator()

    @service(workers=[{"gpus": [0, 1]}, {"gpus": [1, 2]}])
    class Foo:
        pass

    @service(workers=[{"gpus": [0, 5]}])
    class Bar:
        pass

    Foo.inject_config()
    num_workers, worker_env = s.get_worker_env(Foo)
    assert num_workers == 2
    assert worker_env[0]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert worker_env[1]["CUDA_VISIBLE_DEVICES"] == "1,2"
    Bar.inject_config()
    with pytest.raises(ResourceUnavailable):
        num_workers, worker_env = s.get_worker_env(Bar)
