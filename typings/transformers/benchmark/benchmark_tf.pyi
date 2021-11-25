

import tensorflow as tf

from ..configuration_utils import PretrainedConfig
from ..file_utils import is_py3nvml_available, is_tf_available
from .benchmark_args_tf import TensorFlowBenchmarkArguments
from .benchmark_utils import Benchmark

"""
    Benchmarking the library on inference and training in PyTorch.
"""
if is_tf_available():
    ...
if is_py3nvml_available():
    ...
logger = ...
def run_with_tf_optimizations(do_eager_mode: bool, use_xla: bool): # -> (func: Unknown) -> (*args: Unknown, **kwargs: Unknown) -> Unknown:
    ...

def random_input_ids(batch_size: int, sequence_length: int, vocab_size: int) -> [tf.Tensor]:
    ...

class TensorFlowBenchmark(Benchmark):
    args: TensorFlowBenchmarkArguments
    configs: PretrainedConfig
    framework: str = ...
    @property
    def framework_version(self):
        ...
    


