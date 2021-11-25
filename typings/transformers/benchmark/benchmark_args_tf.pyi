

from dataclasses import dataclass

import tensorflow as tf

from ..file_utils import is_tf_available, tf_required
from .benchmark_args_utils import BenchmarkArguments

if is_tf_available():
    ...
logger = ...
@dataclass
class TensorFlowBenchmarkArguments(BenchmarkArguments):
    deprecated_args = ...
    def __init__(self, **kwargs) -> None:
        """
        This __init__ is there for legacy code. When removing deprecated args completely, the class can simply be
        deleted
        """
        ...
    
    tpu_name: str = ...
    device_idx: int = ...
    eager_mode: bool = ...
    use_xla: bool = ...
    @property
    @tf_required
    def is_tpu(self) -> bool:
        ...
    
    @property
    @tf_required
    def strategy(self) -> tf.distribute.Strategy:
        ...
    
    @property
    @tf_required
    def gpu_list(self):
        ...
    
    @property
    @tf_required
    def n_gpu(self) -> int:
        ...
    
    @property
    def is_gpu(self) -> bool:
        ...
    


