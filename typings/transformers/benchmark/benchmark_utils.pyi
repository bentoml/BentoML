

import platform
from abc import ABC, abstractmethod
from typing import Callable, Iterable, List, NamedTuple, Optional, Union

from .. import PretrainedConfig
from ..file_utils import (
    is_psutil_available,
    is_py3nvml_available,
    is_tf_available,
    is_torch_available,
)
from .benchmark_args_utils import BenchmarkArguments

"""
Utilities for working with the local dataset cache.
"""
if is_torch_available():
    ...
if is_tf_available():
    ...
if is_psutil_available():
    ...
if is_py3nvml_available():
    ...
if platform.system() == "Windows":
    ...
else:
    ...
logger = ...
_is_memory_tracing_enabled = ...
BenchmarkOutput = ...
def separate_process_wrapper_fn(func: Callable[[], None], do_multi_processing: bool) -> Callable[[], None]:
    """
    This function wraps another function into its own separated process. In order to ensure accurate memory
    measurements it is important that the function is executed in a separate process

    Args:

        - `func`: (`callable`): function() -> ... generic function which will be executed in its own separate process
        - `do_multi_processing`: (`bool`) Whether to run function on separate process or not
    """
    ...

def is_memory_tracing_enabled(): # -> Literal[False]:
    ...

class Frame(NamedTuple):
    """
    `Frame` is a NamedTuple used to gather the current frame state. `Frame` has the following fields:

        - 'filename' (string): Name of the file currently executed
        - 'module' (string): Name of the module currently executed
        - 'line_number' (int): Number of the line currently executed
        - 'event' (string): Event that triggered the tracing (default will be "line")
        - 'line_text' (string): Text of the line in the python script
    """
    filename: str
    module: str
    line_number: int
    event: str
    line_text: str
    ...


class UsedMemoryState(NamedTuple):
    """
    `UsedMemoryState` are named tuples with the following fields:

        - 'frame': a `Frame` namedtuple (see below) storing information on the current tracing frame (current file,
          location in current file)
        - 'cpu_memory': CPU RSS memory state *before* executing the line
        - 'gpu_memory': GPU used memory *before* executing the line (sum for all GPUs or for only `gpus_to_trace` if
          provided)
    """
    frame: Frame
    cpu_memory: int
    gpu_memory: int
    ...


class Memory(NamedTuple):
    """
    `Memory` NamedTuple have a single field `bytes` and you can get a human readable str of the number of mega bytes by
    calling `__repr__`

        - `byte` (integer): number of bytes,
    """
    bytes: int
    def __repr__(self) -> str:
        ...
    


class MemoryState(NamedTuple):
    """
    `MemoryState` are namedtuples listing frame + CPU/GPU memory with the following fields:

        - `frame` (`Frame`): the current frame (see above)
        - `cpu`: CPU memory consumed at during the current frame as a `Memory` named tuple
        - `gpu`: GPU memory consumed at during the current frame as a `Memory` named tuple
        - `cpu_gpu`: CPU + GPU memory consumed at during the current frame as a `Memory` named tuple
    """
    frame: Frame
    cpu: Memory
    gpu: Memory
    cpu_gpu: Memory
    ...


class MemorySummary(NamedTuple):
    """
    `MemorySummary` namedtuple otherwise with the fields:

        - `sequential`: a list of `MemoryState` namedtuple (see below) computed from the provided `memory_trace` by
          subtracting the memory after executing each line from the memory before executing said line.
        - `cumulative`: a list of `MemoryState` namedtuple (see below) with cumulative increase in memory for each line
          obtained by summing repeated memory increase for a line if it's executed several times. The list is sorted
          from the frame with the largest memory consumption to the frame with the smallest (can be negative if memory
          is released)
        - `total`: total memory increase during the full tracing as a `Memory` named tuple (see below). Line with
          memory release (negative consumption) are ignored if `ignore_released_memory` is `True` (default).
    """
    sequential: List[MemoryState]
    cumulative: List[MemoryState]
    current: List[MemoryState]
    total: Memory
    ...


MemoryTrace = List[UsedMemoryState]
def measure_peak_memory_cpu(function: Callable[[], None], interval=..., device_idx=...) -> int:
    """
    measures peak cpu memory consumption of a given `function` running the function for at least interval seconds and
    at most 20 * interval seconds. This function is heavily inspired by: `memory_usage` of the package
    `memory_profiler`:
    https://github.com/pythonprofilers/memory_profiler/blob/895c4ac7a08020d66ae001e24067da6dcea42451/memory_profiler.py#L239

    Args:

        - `function`: (`callable`): function() -> ... function without any arguments to measure for which to measure
          the peak memory

        - `interval`: (`float`, `optional`, defaults to `0.5`) interval in second for which to measure the memory usage

        - `device_idx`: (`int`, `optional`, defaults to `None`) device id for which to measure gpu usage

    Returns:

        - `max_memory`: (`int`) consumed memory peak in Bytes
    """
    ...

def start_memory_tracing(modules_to_trace: Optional[Union[str, Iterable[str]]] = ..., modules_not_to_trace: Optional[Union[str, Iterable[str]]] = ..., events_to_trace: str = ..., gpus_to_trace: Optional[List[int]] = ...) -> MemoryTrace:
    """
    Setup line-by-line tracing to record rss mem (RAM) at each line of a module or sub-module. See `./benchmark.py` for
    usage examples. Current memory consumption is returned using psutil and in particular is the RSS memory "Resident
    Set Size” (the non-swapped physical memory the process is using). See
    https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_info

    Args:

        - `modules_to_trace`: (None, string, list/tuple of string) if None, all events are recorded if string or list
          of strings: only events from the listed module/sub-module will be recorded (e.g. 'fairseq' or
          'transformers.models.gpt2.modeling_gpt2')
        - `modules_not_to_trace`: (None, string, list/tuple of string) if None, no module is avoided if string or list
          of strings: events from the listed module/sub-module will not be recorded (e.g. 'torch')
        - `events_to_trace`: string or list of string of events to be recorded (see official python doc for
          `sys.settrace` for the list of events) default to line
        - `gpus_to_trace`: (optional list, default None) list of GPUs to trace. Default to tracing all GPUs

    Return:

        - `memory_trace` is a list of `UsedMemoryState` for each event (default each line of the traced script).

            - `UsedMemoryState` are named tuples with the following fields:

                - 'frame': a `Frame` namedtuple (see below) storing information on the current tracing frame (current
                  file, location in current file)
                - 'cpu_memory': CPU RSS memory state *before* executing the line
                - 'gpu_memory': GPU used memory *before* executing the line (sum for all GPUs or for only
                  `gpus_to_trace` if provided)

    `Frame` is a namedtuple used by `UsedMemoryState` to list the current frame state. `Frame` has the following
    fields: - 'filename' (string): Name of the file currently executed - 'module' (string): Name of the module
    currently executed - 'line_number' (int): Number of the line currently executed - 'event' (string): Event that
    triggered the tracing (default will be "line") - 'line_text' (string): Text of the line in the python script

    """
    ...

def stop_memory_tracing(memory_trace: Optional[MemoryTrace] = ..., ignore_released_memory: bool = ...) -> Optional[MemorySummary]:
    """
    Stop memory tracing cleanly and return a summary of the memory trace if a trace is given.

    Args:

        `memory_trace` (optional output of start_memory_tracing, default: None):
            memory trace to convert in summary
        `ignore_released_memory` (boolean, default: None):
            if True we only sum memory increase to compute total memory

    Return:

        - None if `memory_trace` is None
        - `MemorySummary` namedtuple otherwise with the fields:

            - `sequential`: a list of `MemoryState` namedtuple (see below) computed from the provided `memory_trace` by
              subtracting the memory after executing each line from the memory before executing said line.
            - `cumulative`: a list of `MemoryState` namedtuple (see below) with cumulative increase in memory for each
              line obtained by summing repeated memory increase for a line if it's executed several times. The list is
              sorted from the frame with the largest memory consumption to the frame with the smallest (can be negative
              if memory is released)
            - `total`: total memory increase during the full tracing as a `Memory` named tuple (see below). Line with
              memory release (negative consumption) are ignored if `ignore_released_memory` is `True` (default).

    `Memory` named tuple have fields

        - `byte` (integer): number of bytes,
        - `string` (string): same as human readable string (ex: "3.5MB")

    `Frame` are namedtuple used to list the current frame state and have the following fields:

        - 'filename' (string): Name of the file currently executed
        - 'module' (string): Name of the module currently executed
        - 'line_number' (int): Number of the line currently executed
        - 'event' (string): Event that triggered the tracing (default will be "line")
        - 'line_text' (string): Text of the line in the python script

    `MemoryState` are namedtuples listing frame + CPU/GPU memory with the following fields:

        - `frame` (`Frame`): the current frame (see above)
        - `cpu`: CPU memory consumed at during the current frame as a `Memory` named tuple
        - `gpu`: GPU memory consumed at during the current frame as a `Memory` named tuple
        - `cpu_gpu`: CPU + GPU memory consumed at during the current frame as a `Memory` named tuple
    """
    ...

def bytes_to_mega_bytes(memory_amount: int) -> int:
    """Utility to convert a number of bytes (int) into a number of mega bytes (int)"""
    ...

class Benchmark(ABC):
    """
    Benchmarks is a simple but feature-complete benchmarking script to compare memory and time performance of models in
    Transformers.
    """
    args: BenchmarkArguments
    configs: PretrainedConfig
    framework: str
    def __init__(self, args: BenchmarkArguments = ..., configs: PretrainedConfig = ...) -> None:
        ...
    
    @property
    def print_fn(self): # -> (*args: Unknown) -> None | (*values: object, sep: str | None = ..., end: str | None = ..., file: SupportsWrite[str] | None = ..., flush: bool = ...) -> None:
        ...
    
    @property
    @abstractmethod
    def framework_version(self): # -> None:
        ...
    
    def inference_speed(self, *args, **kwargs) -> float:
        ...
    
    def train_speed(self, *args, **kwargs) -> float:
        ...
    
    def inference_memory(self, *args, **kwargs) -> [Memory, Optional[MemorySummary]]:
        ...
    
    def train_memory(self, *args, **kwargs) -> [Memory, Optional[MemorySummary]]:
        ...
    
    def run(self): # -> BenchmarkOutput:
        ...
    
    @property
    def environment_info(self): # -> dict[Unknown, Unknown]:
        ...
    
    def print_results(self, result_dict, type_label):
        ...
    
    def print_memory_trace_statistics(self, summary: MemorySummary): # -> None:
        ...
    
    def save_to_csv(self, result_dict, filename):
        ...
    


