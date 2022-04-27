from __future__ import annotations

import os
import re
import math
import typing as t
import logging
import functools

import attr
import psutil

from ...exceptions import BentoMLException

logger = logging.getLogger(__name__)


def cpu_converter(cpu: t.Union[int, float, str]) -> float:
    assert isinstance(cpu, (int, float, str)), "cpu must be int, float or str"

    if isinstance(cpu, (int, float)):
        return float(cpu)

    milli_match = re.match("([0-9]+)m", cpu)
    if milli_match:
        return float(milli_match[1]) / 1000.0
    raise BentoMLException(f"Invalid CPU resource limit '{cpu}'. ")


def mem_converter(mem: t.Union[int, str]) -> int:
    assert isinstance(mem, (int, str)), "mem must be int or str"

    if isinstance(mem, int):
        return mem

    unit_match = re.match("([0-9]+)([A-Za-z]{1,2})", mem)
    mem_multipliers = {
        "k": 1000,
        "M": 1000**2,
        "G": 1000**3,
        "T": 1000**4,
        "P": 1000**5,
        "E": 1000**6,
        "Ki": 1024,
        "Mi": 1024**2,
        "Gi": 1024**3,
        "Ti": 1024**4,
        "Pi": 1024**5,
        "Ei": 1024**6,
    }
    if unit_match:
        base = int(unit_match[1])
        unit = unit_match[2]
        if unit in mem_multipliers:
            return base * mem_multipliers[unit]
    raise ValueError(f"Invalid MEM resource limit '{mem}'")


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
    shares_file = os.path.join(cgroup_root, "cpu", "cpu.shares")
    cpu_max_file = os.path.join(cgroup_root, "cpu.max")

    quota, shares = None, None

    if os.path.exists(cfs_quota_us_file) and os.path.exists(cfs_period_us_file):
        try:
            quota = _read_cgroup_file(cfs_quota_us_file) / _read_cgroup_file(
                cfs_period_us_file
            )
        except FileNotFoundError as err:
            logger.warning(f"Caught exception while calculating CPU quota: {err}")
    # reading from cpu.max for cgroup v2
    elif os.path.exists(cpu_max_file):
        try:
            with open(cpu_max_file, "r") as max_file:
                cfs_string = max_file.read()
                quota_str, period_str = cfs_string.split()
                if quota_str.isnumeric() and period_str.isnumeric():
                    quota = float(quota_str) / float(period_str)
                else:
                    # quota_str is "max" meaning the cpu quota is unset
                    quota = None
        except FileNotFoundError as err:
            logger.warning(f"Caught exception while calculating CPU quota: {err}")
    if quota is not None and quota < 0:
        quota = None
    elif quota == 0:
        quota = 1

    if os.path.exists(shares_file):
        try:
            shares = _read_cgroup_file(shares_file) / float(1024)
        except FileNotFoundError as err:
            logger.warning(f"Caught exception while getting CPU shares: {err}")

    os_cpu_count = float(os.cpu_count() or 1.0)

    limit_count = math.inf

    if quota and shares:
        limit_count = min(quota, shares)
    elif quota:
        limit_count = quota
    elif shares:
        limit_count = shares

    return float(min(limit_count, os_cpu_count))


def query_os_cpu_count() -> int:
    cpu_count = os.cpu_count()
    if cpu_count is not None:
        return cpu_count
    logger.warning("os.cpu_count() is NoneType")
    return 1


def query_cpu_count() -> float | int:
    # Default to the total CPU count available in current node or cgroup
    if psutil.POSIX:
        return query_cgroup_cpu_count()
    else:
        return query_os_cpu_count()


@functools.lru_cache(maxsize=1)
def query_nvidia_gpu_count() -> int:
    """
    query nvidia gpu count, available on Windows and Linux
    """
    import pynvml.nvml  # type: ignore
    from pynvml.smi import nvidia_smi  # type: ignore

    try:
        inst = nvidia_smi.getInstance()
        query: t.Dict[str, int] = inst.DeviceQuery("count")  # type: ignore
        return query.get("count", 0)
    except (pynvml.nvml.NVMLError, OSError):
        return 0


def gpu_converter(gpus: t.Optional[t.Union[int, str, t.List[str]]]) -> int:
    if isinstance(gpus, int):
        return gpus

    if isinstance(gpus, str):
        return int(gpus)

    if isinstance(gpus, list):
        return len(gpus)

    return query_nvidia_gpu_count()


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


def _get_default_cpu() -> float:
    # Default to the total CPU count available in current node or cgroup
    if psutil.POSIX:
        return query_cgroup_cpu_count()
    else:
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            return float(cpu_count)
        raise ValueError("CPU count is NoneType")


@attr.define(frozen=True)
class Resource:
    cpu: int = attr.field()
    nvidia_gpu: int = attr.field()
    custom_resources: t.Dict[str, t.Union[float, int]] = attr.field(factory=dict)

    @classmethod
    def from_dict(cls, init_dict: t.Dict[str, t.Any]) -> Resource:
        ...
