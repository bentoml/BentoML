import os
import re
from functools import lru_cache
from typing import Union


def _cpu_converter(cpu: Union[int, float, str]) -> float:
    if isinstance(cpu, (int, float)):
        return float(cpu)

    if isinstance(cpu, str):
        milli_match = re.match("([0-9]+)m", cpu)
        if milli_match:
            return int(milli_match[1]) / 1000.0

    raise ValueError(f"Invalid CPU resource limit '{cpu}'")


def _mem_converter(mem: Union[int, str]) -> int:
    if isinstance(mem, int):
        return mem

    if isinstance(mem, str):
        unit_match = re.match("([0-9]+)([A-Za-z]{1,2})", mem)
        mem_multipliers = {
            "k": 1000,
            "M": 1000 ** 2,
            "G": 1000 ** 3,
            "T": 1000 ** 4,
            "P": 1000 ** 5,
            "E": 1000 ** 6,
            "Ki": 1024,
            "Mi": 1024 ** 2,
            "Gi": 1024 ** 3,
            "Ti": 1024 ** 4,
            "Pi": 1024 ** 5,
            "Ei": 1024 ** 6,
        }
        if unit_match:
            base = int(unit_match[1])
            unit = unit_match[2]
            if unit in mem_multipliers:
                return base * mem_multipliers[unit]

    raise ValueError(f"Invalid MEM resource limit '{mem}'")


@lru_cache(maxsize=1)
def _query_cpu_count() -> float:
    # Query active cpu processor count using cgroup v1 API, based on OpenJDK
    # implementation for `active_processor_count` using cgroup v1:
    # https://github.com/openjdk/jdk/blob/master/src/hotspot/os/linux/cgroupSubsystem_linux.cpp
    # For cgroup v2, see:
    # https://github.com/openjdk/jdk/blob/master/src/hotspot/os/linux/cgroupV2Subsystem_linux.cpp
    def _read_integer_file(filename):
        with open(filename, "r") as f:
            return int(f.read().rstrip())

    cgroup_root = "/sys/fs/cgroup/"
    cfs_quota_us_file = os.path.join(cgroup_root, "cpu", "cpu.cfs_quota_us")
    cfs_period_us_file = os.path.join(cgroup_root, "cpu", "cpu.cfs_period_us")
    shares_file = os.path.join(cgroup_root, "cpu", "cpu.shares")

    quota = shares = period = -1
    if os.path.isfile(cfs_quota_us_file):
        quota = _read_integer_file(cfs_quota_us_file)

    if os.path.isfile(shares_file):
        shares = _read_integer_file(shares_file)
        if shares == 1024:
            shares = -1

    if os.path.isfile(cfs_period_us_file):
        period = _read_integer_file(cfs_period_us_file)

    limit_count = cpu_count = os.cpu_count()
    quota_count = 0
    share_count = 0

    if quota > -1 and period > 0:
        quota_count = float(quota) / float(period)
    if shares > -1:
        share_count = float(shares) / float(1024)

    if quota_count != 0 and share_count != 0:
        limit_count = min(quota_count, share_count)
    if quota_count != 0:
        limit_count = quota_count
    if share_count != 0:
        limit_count = share_count

    return float(min(limit_count, cpu_count))
