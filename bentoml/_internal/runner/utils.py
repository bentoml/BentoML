import ctypes
import itertools
import logging
import os
import re
import typing as t
from functools import lru_cache
from typing import TYPE_CHECKING

from ...exceptions import BentoMLException

logger = logging.getLogger(__name__)

# Some constants taken from cuda.h


if TYPE_CHECKING:
    from aiohttp import MultipartWriter

T = t.TypeVar("T")
To = t.TypeVar("To")


CUDA_SUCCESS = 0


class Params(t.Generic[T]):
    def __init__(self, *args: T, **kwargs: T):
        self.args = args
        self.kwargs = kwargs

    def map(self, function: t.Callable[[T], To]) -> "Params[To]":
        args = tuple(function(a) for a in self.args)
        kwargs = {k: function(v) for k, v in self.kwargs.items()}
        return type(self)[To](*args, **kwargs)

    def imap(
        self, function: t.Callable[[T], t.Iterable[To]]
    ) -> "t.Iterator[Params[To]]":
        args_iter = tuple(iter(function(a)) for a in self.args)
        kwargs_iter = {k: iter(function(v)) for k, v in self.kwargs.items()}

        try:
            while True:
                args = tuple(next(a) for a in args_iter)
                kwargs = {k: next(v) for k, v in kwargs_iter.items()}
                yield type(self)[To](*args, **kwargs)
        except StopIteration:
            pass

    def to_http_multipart(self) -> "MultipartWriter":
        d = self.to_dict()
        pass

    def to_dict(self) -> t.Dict[t.Union[int, str], T]:
        d = dict(**self.kwargs)
        for i, v in enumerate(self.args):
            d[i] = v
        return d

    @classmethod
    def from_dict(cls, d: t.Dict[t.Union[int, str], To]) -> "Params[To]":
        args = tuple(
            v for _, v in sorted((k, v) for k, v in d.items() if isinstance(k, int))
        )
        kwargs = {k: v for k, v in d.items() if not isinstance(k, int)}
        return cls[To](*args, **kwargs)

    @property
    def sample(self) -> T:
        if self.args:
            return self.args[0]
        return next(iter(self.kwargs.values()))


class TypeRef:
    def __init__(self, module: str, qualname: str):
        self.module = module
        self.qualname = qualname

    @classmethod
    def from_instance(cls, instance: object) -> "TypeRef":
        klass = type(instance)
        return cls.from_type(klass)

    @classmethod
    def from_type(cls, klass: t.Union["TypeRef", type]) -> "TypeRef":
        if isinstance(klass, type):
            return cls(klass.__module__, klass.__qualname__)
        return klass

    def evaluate(self) -> type:
        import importlib

        m = importlib.import_module(self.module)
        ref = t.ForwardRef(f"m.{self.qualname}")
        localns = {"m": m}

        if hasattr(t, "_eval_type"):  # python3.8 & python 3.9
            _eval_type = getattr(t, "_eval_type")
            return _eval_type(ref, globals(), localns)

        if hasattr(ref, "_eval_type"):  # python3.6
            _eval_type = getattr(ref, "_eval_type")
            return _eval_type(globals(), localns)

        raise SystemError("unsupported Python version")

    def __eq__(self, o: t.Union["TypeRef", type]) -> bool:
        '''
        TypeRef("numpy", "ndarray") == np.ndarray
        TypeRef("numpy", "ndarray") == TypeRef.from_instance(np.random.randint([2, 3]))
        '''
        if isinstance(o, type):
            o = self.from_type(type)
        return self.module == o.module and self.qualname == o.qualname

    def __hash__(self):
        return hash(f"{self.module}.{self.qualname}")


def _cpu_converter(cpu: t.Union[int, float, str]) -> float:
    if isinstance(cpu, (int, float)):
        return float(cpu)

    if isinstance(cpu, str):
        milli_match = re.match("([0-9]+)m", cpu)
        if milli_match:
            return int(milli_match[1]) / 1000.0

    raise ValueError(f"Invalid CPU resource limit '{cpu}'")


def _mem_converter(mem: t.Union[int, str]) -> int:
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
def _query_cgroup_cpu_count() -> float:
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


@lru_cache(maxsize=1)
def _cuda_lib() -> "ctypes.CDLL":
    libs = ("libcuda.so", "cuda.dll")
    for lib in libs:
        try:
            return ctypes.CDLL(lib)
        except OSError:
            continue
    else:
        raise OSError(f"could not load any of: {' '.join(libs)}")


def _gpu_converter(gpus: t.Optional[t.Union[int, str, t.List[str]]]) -> t.List[str]:
    # https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html
    if gpus is not None:
        err = ctypes.c_char_p()
        device = ctypes.c_int()
        num_gpus = ctypes.c_int()

        try:
            drv = _cuda_lib()
            res = drv.cuInit(0)
            if res != CUDA_SUCCESS:
                drv.cuGetErrorString(res, ctypes.byref(err))
                logger.error(
                    f"cuInit failed with error code {res}: {err.value.decode()}"
                )

            res = drv.cuDeviceGetCount(ctypes.byref(num_gpus))
            if res != CUDA_SUCCESS:
                drv.cuGetErrorString(res, ctypes.byref(err))
                logger.error(
                    "cuDeviceGetCount failed "
                    f"with error code {res}: {err.value.decode()}"
                )

            def _validate_dev(dev_id: t.Union[int, str]) -> bool:
                _res = drv.cuDeviceGet(ctypes.byref(device), int(dev_id))
                if _res != CUDA_SUCCESS:
                    drv.cuGetErrorString(res, ctypes.byref(err))
                    logger.warning(
                        "cuDeviceGet failed "
                        f"with error code {res}: {err.value.decode()}"
                    )
                    return False
                return True

            if any([isinstance(gpus, i) for i in [str, int]]):
                if gpus == "all":
                    return [str(dev) for dev in range(num_gpus.value)]
                else:
                    if _validate_dev(gpus):
                        return [str(gpus)]
                    raise BentoMLException(
                        f"Unknown GPU devices. Available devices: {num_gpus.value}"
                    )
            else:
                return list(
                    itertools.chain.from_iterable([_gpu_converter(gpu) for gpu in gpus])
                )
        except OSError as e:
            raise BentoMLException(
                f"{e}\nMake sure to have CUDA "
                f"installed you are intending "
                f"to use GPUs with BentoML."
            )
    return list()
