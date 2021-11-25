

from dataclasses import dataclass

import torch

from ..file_utils import is_torch_available, is_torch_tpu_available, torch_required
from .benchmark_args_utils import BenchmarkArguments

if is_torch_available():
    ...
if is_torch_tpu_available():
    ...
logger = ...
@dataclass
class PyTorchBenchmarkArguments(BenchmarkArguments):
    deprecated_args = ...
    def __init__(self, **kwargs) -> None:
        """
        This __init__ is there for legacy code. When removing deprecated args completely, the class can simply be
        deleted
        """
        ...
    
    torchscript: bool = ...
    torch_xla_tpu_print_metrics: bool = ...
    fp16_opt_level: str = ...
    @property
    def is_tpu(self): # -> bool:
        ...
    
    @property
    @torch_required
    def device_idx(self) -> int:
        ...
    
    @property
    @torch_required
    def device(self) -> torch.device:
        ...
    
    @property
    @torch_required
    def n_gpu(self): # -> int:
        ...
    
    @property
    def is_gpu(self): # -> bool:
        ...
    


