from __future__ import annotations

import os
import re
import math
import typing as t
import logging
import functools
from abc import ABC
from abc import abstractmethod

import psutil

from ..exceptions import BentoMLConfigException

logger = logging.getLogger(__name__)

_RESOURCE_REGISTRY: dict[str, t.Type[Resource[t.Any]]] = {}

T = t.TypeVar("T")


def get_resource(
    resources: dict[str, t.Any], resource_kind: str, validate: bool = True
) -> t.Any:
    if resource_kind not in _RESOURCE_REGISTRY:
        raise BentoMLConfigException(f"Unknown resource kind '{resource_kind}'.")

    resource: t.Type[Resource[t.Any]] = _RESOURCE_REGISTRY[resource_kind]

    if resource_kind in resources:
        if resources[resource_kind] == "system":
            return resource.from_system()
        else:
            res = resource.from_spec(resources[resource_kind])
            if validate:
                resource.validate(res)
            return res
    else:
        return None


def system_resources() -> dict[str, t.Any]:
    res: dict[str, t.Any] = {}
    for resource_kind, resource in _RESOURCE_REGISTRY.items():
        res[resource_kind] = resource.from_system()
    return res


class Resource(t.Generic[T], ABC):
    def __init_subclass__(cls, *, resource_id: str):  # pylint: disable=arguments-differ
        _RESOURCE_REGISTRY[resource_id] = cls

    @classmethod
    @abstractmethod
    def from_spec(cls, spec: t.Any) -> T:
        """
        Get an instance of this resource from user input. For example, a CPU resource
        might parse "10m" and return a CPU resource with 0.01 CPUs.
        """

    @classmethod
    @abstractmethod
    def from_system(cls) -> T:
        """
        Infer resource value from the system.
        """

    @classmethod
    @abstractmethod
    def validate(cls, val: T):
        """
        Validate that the resources are available on the current system.
        """


class CpuResource(Resource[float], resource_id="cpu"):
    @classmethod
    def from_spec(cls, spec: t.Any) -> float:
        """
        Convert spec to CpuResource.

        spec can be a float, int or string.
        - 1.0 -> 1.0
        - 1 -> 1.0
        - "1" -> 1.0
        - "10m" -> 0.01
        """
        if not isinstance(spec, (int, float, str)):
            raise TypeError("cpu must be int, float or str")

        if isinstance(spec, (int, float)):
            return float(spec)

        milli_match = re.match("([0-9]+)m", spec)
        if milli_match:
            return float(milli_match[1]) / 1000.0

        try:
            return float(spec)
        except ValueError:
            raise BentoMLConfigException(f"Invalid CPU resource limit '{spec}'. ")

    @classmethod
    def from_system(cls) -> float:
        if psutil.POSIX:
            return query_cgroup_cpu_count()
        else:
            return float(query_os_cpu_count())

    @classmethod
    def validate(cls, val: float):
        if val < 0:
            raise BentoMLConfigException(
                f"Invalid negative CPU resource limit '{val}'."
            )
        if not math.isclose(val, cls.from_system()) and val > cls.from_system():
            raise BentoMLConfigException(
                f"CPU resource limit {val} is greater than the system available: {cls.from_system()}"
            )


@functools.lru_cache(maxsize=1)
def query_cgroup_cpu_count() -> float:
    # Query active cpu processor count using cgroup v1 API, based on OpenJDK
    # implementation for `active_processor_count` using cgroup v1:
    # https://github.com/openjdk/jdk/blob/master/src/hotspot/os/linux/cgroupSubsystem_linux.cpp
    # For cgroup v2, see:
    # https://github.com/openjdk/jdk/blob/master/src/hotspot/os/linux/cgroupV2Subsystem_linux.cpp
    # Possible supports: cpuset.cpus on kubernetes
    def _read_cgroup_file(filename: str) -> float:
        with open(filename, "r", encoding="utf-8") as f:
            return int(f.read().strip())

    cgroup_root = "/sys/fs/cgroup/"
    cfs_quota_us_file = os.path.join(cgroup_root, "cpu", "cpu.cfs_quota_us")
    cfs_period_us_file = os.path.join(cgroup_root, "cpu", "cpu.cfs_period_us")
    cpu_max_file = os.path.join(cgroup_root, "cpu.max")

    quota = None

    if os.path.exists(cfs_quota_us_file) and os.path.exists(cfs_period_us_file):
        try:
            quota = _read_cgroup_file(cfs_quota_us_file) / _read_cgroup_file(
                cfs_period_us_file
            )
        except FileNotFoundError as err:
            logger.warning("Caught exception while calculating CPU quota: %s", err)
    # reading from cpu.max for cgroup v2
    elif os.path.exists(cpu_max_file):
        try:
            with open(cpu_max_file, "r", encoding="utf-8") as max_file:
                cfs_string = max_file.read()
                quota_str, period_str = cfs_string.split()
                if quota_str.isnumeric() and period_str.isnumeric():
                    quota = float(quota_str) / float(period_str)
                else:
                    # quota_str is "max" meaning the cpu quota is unset
                    quota = None
        except FileNotFoundError as err:
            logger.warning("Caught exception while calculating CPU quota: %s", err)
    if quota is not None and quota < 0:
        quota = None
    elif quota == 0:
        quota = 1

    os_cpu_count = float(os.cpu_count() or 1.0)

    limit_count = math.inf

    if quota:
        limit_count = quota

    return float(min(limit_count, os_cpu_count))


@functools.lru_cache(maxsize=1)
def query_os_cpu_count() -> int:
    cpu_count = os.cpu_count()
    if cpu_count is not None:
        return cpu_count
    logger.warning("Failed to determine CPU count, using 1 as default.")
    return 1


# class MemResource(Resource, resource_id="mem"):
#     @classmethod
#     def from_spec(cls, spec: t.Any):
#         assert isinstance(mem, (int, str)), "mem must be int or str"
#
#         if isinstance(mem, int):
#             return mem
#
#         unit_match = re.match("([0-9]+)([A-Za-z]{1,2})", mem)
#         mem_multipliers = {
#             "k": 1000,
#             "M": 1000**2,
#             "G": 1000**3,
#             "T": 1000**4,
#             "P": 1000**5,
#             "E": 1000**6,
#             "Ki": 1024,
#             "Mi": 1024**2,
#             "Gi": 1024**3,
#             "Ti": 1024**4,
#             "Pi": 1024**5,
#             "Ei": 1024**6,
#         }
#         if unit_match:
#             base = int(unit_match[1])
#             unit = unit_match[2]
#             if unit in mem_multipliers:
#                 return base * mem_multipliers[unit]
#         raise ValueError(f"Invalid MEM resource limit '{mem}'")


class NvidiaGpuResource(Resource[t.List[int]], resource_id="nvidia.com/gpu"):
    @classmethod
    def from_spec(cls, spec: int | str | list[int | str]) -> list[int]:
        if not isinstance(spec, (int, str, list)):
            raise TypeError(
                "NVidia GPU device IDs must be int, str or a list specifing the exact GPUs to use."
            )

        try:
            if isinstance(spec, int):
                if spec < -1:
                    raise ValueError
                return list(range(spec))
            elif isinstance(spec, str):
                return cls.from_spec(int(spec))
            else:
                return [int(x) for x in spec]
        except ValueError:
            raise BentoMLConfigException(
                f"Invalid NVidia GPU resource limit '{spec}'. "
            )

    @classmethod
    @functools.lru_cache(maxsize=1)
    def from_system(cls) -> list[int]:
        """
        query nvidia gpu count, available on Windows and Linux
        """
        import pynvml  # type: ignore

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            return list(range(device_count))
        except (pynvml.nvml.NVMLError, OSError):
            logger.debug("GPU not detected. Unable to initialize pynvml lib.")
            return []
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # pylint: disable=broad-except
                pass

    @classmethod
    def validate(cls, val: t.List[int]):
        if any([gpu_index < 0 for gpu_index in val]):
            raise BentoMLConfigException(f"Negative GPU device in {val}.")
        if any([gpu_index >= len(cls.from_system()) for gpu_index in val]):
            raise BentoMLConfigException(
                f"GPU device index in {val} is greater than the system available: {cls.from_system()}"
            )


def get_gpu_memory(dev: int) -> t.Tuple[float, float]:
    """
    Return Total Memory and Free Memory in given GPU device. in MiB
    """
    import pynvml.nvml  # type: ignore
    from pynvml.smi import nvidia_smi  # type: ignore

    unit_multiplier = {
        "PiB": 1024.0 * 1024 * 1024,
        "TiB": 1024.0 * 1024,
        "GiB": 1024.0,
        "MiB": 1.0,
        "KiB": 1.0 / 1024,
        "B": 1.0 / 1024 / 1024,
    }

    try:
        inst = nvidia_smi.getInstance()
        query: t.Dict[str, int] = inst.DeviceQuery(dev)  # type: ignore
    except (pynvml.nvml.NVMLError, OSError):
        return 0.0, 0.0

    try:
        gpus: t.List[t.Dict[str, t.Any]] = query.get("gpu", [])  # type: ignore
        gpu = gpus[dev]
        unit = gpu["fb_memory_usage"]["unit"]
        total = gpu["fb_memory_usage"]["total"] * unit_multiplier[unit]
        free = gpu["fb_memory_usage"]["free"] * unit_multiplier[unit]
        return total, free
    except IndexError:
        raise ValueError(f"Invalid GPU device index {dev}")
    except KeyError:
        raise RuntimeError(f"unexpected nvml query result: {query}")
