import os
import re
import math
import typing as t
import logging
import itertools
from typing import TYPE_CHECKING
from functools import lru_cache

from ...exceptions import BentoMLException

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from aiohttp import MultipartWriter
    from starlette.requests import Request

    from ..runner.container import Payload


T = t.TypeVar("T")
To = t.TypeVar("To")


CUDA_SUCCESS = 0


class Params(t.Generic[T]):
    def __init__(self, *args: T, **kwargs: T):
        self.args: t.Tuple[T, ...] = args
        self.kwargs: t.Dict[str, T] = kwargs

    def map(self, function: t.Callable[[T], To]) -> "Params[To]":
        args = tuple(function(a) for a in self.args)
        kwargs = {k: function(v) for k, v in self.kwargs.items()}
        return Params[To](*args, **kwargs)

    def imap(
        self, function: t.Callable[[T], t.Iterable[To]]
    ) -> "t.Iterator[Params[To]]":
        args_iter = tuple(iter(function(a)) for a in self.args)
        kwargs_iter = {k: iter(function(v)) for k, v in self.kwargs.items()}

        try:
            while True:
                args = tuple(next(a) for a in args_iter)
                kwargs = {k: next(v) for k, v in kwargs_iter.items()}
                yield Params[To](*args, **kwargs)
        except StopIteration:
            pass

    def items(self) -> t.Iterator[t.Tuple[t.Union[int, str], T]]:
        return itertools.chain(enumerate(self.args), self.kwargs.items())

    @classmethod
    def agg(
        cls,
        params_list: t.Sequence["Params[T]"],
        agg_func: t.Callable[[t.Sequence[T]], To] = lambda i: i,
    ) -> "Params[To]":
        if not params_list:
            return t.cast(Params[To], [])

        args: t.List[To] = []
        kwargs: t.Dict[str, To] = {}

        for j, _ in enumerate(params_list[0].args):
            arg: t.List[T] = []
            for params in params_list:
                arg.append(params.args[j])
            args.append(agg_func(arg))
        for k in params_list[0].kwargs:
            kwarg: t.List[T] = []
            for params in params_list:
                kwarg.append(params.kwargs[k])
            kwargs[k] = agg_func(kwarg)
        return Params(*tuple(args), **kwargs)

    @property
    def sample(self) -> T:
        if self.args:
            return self.args[0]
        return next(iter(self.kwargs.values()))


PAYLOAD_META_HEADER = "Bento-Payload-Meta"


def payload_params_to_multipart(params: Params["Payload"]) -> "MultipartWriter":
    import json

    from multidict import CIMultiDict
    from aiohttp.multipart import MultipartWriter

    multipart = MultipartWriter(subtype="form-data")
    for key, payload in params.items():
        multipart.append(
            payload.data,
            headers=CIMultiDict(
                (
                    (PAYLOAD_META_HEADER, json.dumps(payload.meta)),
                    ("Content-Disposition", f'form-data; name="{key}"'),
                )
            ),
        )
    return multipart


async def multipart_to_payload_params(request: "Request") -> Params["Payload"]:
    import json

    from bentoml._internal.runner.container import Payload
    from bentoml._internal.utils.formparser import populate_multipart_requests

    parts = await populate_multipart_requests(request)
    max_arg_index = -1
    kwargs: t.Dict[str, Payload] = {}
    args_map: t.Dict[int, Payload] = {}
    for field_name, req in parts.items():
        payload = Payload(
            data=await req.body(),
            meta=json.loads(req.headers[PAYLOAD_META_HEADER]),
        )
        if field_name.isdigit():
            arg_index = int(field_name)
            args_map[arg_index] = payload
            max_arg_index = max(max_arg_index, arg_index)
        else:
            kwargs[field_name] = payload
    args = tuple(args_map[i] for i in range(max_arg_index + 1))
    return Params(*args, **kwargs)


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


@lru_cache(maxsize=1)
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


@lru_cache(maxsize=1)
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
