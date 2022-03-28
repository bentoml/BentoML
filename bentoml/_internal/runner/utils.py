import os
import re
import math
import ctypes
import typing as t
import logging
import itertools
from typing import TYPE_CHECKING
from functools import lru_cache

from simple_di.providers import SingletonFactory

from ...exceptions import BentoMLException

logger = logging.getLogger(__name__)

# Some constants taken from cuda.h


if TYPE_CHECKING:
    from ctypes import c_int
    from ctypes import c_char_p
    from ctypes import c_size_t
    from ctypes import c_void_p

    CDataType = t.Union[c_int, c_void_p, c_size_t, c_char_p]

    from aiohttp import MultipartWriter
    from starlette.requests import Request

    from ..runner.container import Payload

    class PlcType(t.TypedDict):
        err: c_char_p
        device: c_int
        num_gpus: c_int
        context: c_void_p
        free_mem: c_size_t
        total_mem: c_size_t


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
    if isinstance(cpu, (int, float)):
        return float(cpu)

    if isinstance(cpu, str):
        milli_match = re.match("([0-9]+)m", cpu)
        if milli_match:
            return int(milli_match[1]) / 1000.0

    raise ValueError(f"Invalid CPU resource limit '{cpu}'")


def mem_converter(mem: t.Union[int, str]) -> int:
    if isinstance(mem, int):
        return mem

    if isinstance(mem, str):
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


@SingletonFactory
def _cuda_lib() -> "ctypes.CDLL":
    libs = ("libcuda.so", "cuda.dll")
    for lib in libs:
        try:
            return ctypes.CDLL(lib)
        except OSError:
            continue
    raise OSError(f"could not load any of: {' '.join(libs)}")


@SingletonFactory
def _init_var() -> t.Tuple["ctypes.CDLL", "PlcType"]:
    # https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html
    # TODO: add threads_per_core, cores, Compute Capability
    err = ctypes.c_char_p()
    plc: PlcType = {
        "err": err,
        "device": ctypes.c_int(),
        "num_gpus": ctypes.c_int(),
        "context": ctypes.c_void_p(),
        "free_mem": ctypes.c_size_t(),
        "total_mem": ctypes.c_size_t(),
    }

    try:
        _drv = _cuda_lib.get()
        res = _drv.cuInit(0)
        if res != CUDA_SUCCESS:
            _drv.cuGetErrorString(res, ctypes.byref(err))
            logger.error(f"cuInit failed with error code {res}: {str(err.value)}")
        return _drv, plc
    except OSError as e:
        raise BentoMLException(
            f"{e}\nMake sure to have CUDA "
            f"installed you are intending "
            f"to use GPUs with BentoML."
        )


def gpu_converter(gpus: t.Optional[t.Union[int, str, t.List[str]]]) -> t.List[str]:
    if gpus is not None:
        drv, plc = _init_var.get()

        res = drv.cuDeviceGetCount(ctypes.byref(plc["num_gpus"]))
        if res != CUDA_SUCCESS:
            drv.cuGetErrorString(res, ctypes.byref(plc["err"]))
            logger.error(
                f"cuDeviceGetCount failed with error code {res}: {str(plc['err'].value)}"
            )

        def _validate_dev(dev_id: t.Union[int, str]) -> bool:
            _res = drv.cuDeviceGet(ctypes.byref(plc["device"]), int(dev_id))
            if _res != CUDA_SUCCESS:
                drv.cuGetErrorString(_res, ctypes.byref(plc["err"]))
                logger.warning(
                    "cuDeviceGet failed "
                    f"with error code {_res}: {str(plc['err'].value)}"
                )
                return False
            return True

        if isinstance(gpus, (int, str)):
            if gpus == "all":
                return [str(dev) for dev in range(plc["num_gpus"].value)]
            else:
                if _validate_dev(gpus):
                    return [str(gpus)]
                raise BentoMLException(
                    f"Unknown GPU devices. Available devices: {plc['num_gpus'].value}"
                )
        else:
            return list(
                itertools.chain.from_iterable([gpu_converter(gpu) for gpu in gpus])
            )
    return list()


def get_gpu_memory(dev: int) -> t.Tuple[int, int]:
    """Return Total Memory and Free Memory in given GPU device. in MiB"""
    drv, plc = _init_var.get()

    res = drv.cuDeviceGet(ctypes.byref(plc["device"]), dev)
    if res != CUDA_SUCCESS:
        drv.cuGetErrorString(res, ctypes.byref(plc["err"]))
        logger.error(
            "cuDeviceGet failed " f"with error code {res}: {str(plc['err'].value)}"
        )
    try:
        res = drv.cuCtxCreate_v2(ctypes.byref(plc["context"]), 0, plc["device"])
    except AttributeError:
        res = drv.cuCtxCreate(ctypes.byref(plc["context"]), 0, plc["device"])
    if res != CUDA_SUCCESS:
        drv.cuGetErrorString(res, ctypes.byref(plc["err"]))
        logger.error(
            f"cuCtxCreate failed with error code {res}: {str(plc['err'].value)}"
        )

    try:
        res = drv.cuMemGetInfo_v2(
            ctypes.byref(plc["free_mem"]), ctypes.byref(plc["total_mem"])
        )
    except AttributeError:
        res = drv.cuMemGetInfo(
            ctypes.byref(plc["free_mem"]), ctypes.byref(plc["total_mem"])
        )
    if res != CUDA_SUCCESS:
        drv.cuGetErrorString(res, ctypes.byref(plc["err"]))
        logger.error(
            f"cuMemGetInfo failed with error code {res}: " f"{str(plc['err'].value)}"
        )
    _total_mem = plc["total_mem"].value
    _free_mem = plc["free_mem"].value
    logger.debug(f"Total Memory: {_total_mem} MiB\nFree Memory: {_free_mem} MiB")
    drv.cuCtxDetach(plc["context"])
    return _total_mem, _free_mem
