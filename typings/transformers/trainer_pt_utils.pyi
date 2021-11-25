

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Union

import smdistributed.modelparallel.torch as smp
import torch
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler

from .file_utils import (
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)

if is_sagemaker_dp_enabled():
    ...
else:
    ...
if is_torch_tpu_available():
    ...
logger = ...
def torch_pad_and_concatenate(tensor1, tensor2, padding_index=...):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    ...

def numpy_pad_and_concatenate(array1, array2, padding_index=...):
    """Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary."""
    ...

def nested_concat(tensors, new_tensors, padding_index=...):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples of tensors.
    """
    ...

def find_batch_size(tensors):
    """
    Find the first dimension of a tensor in a nested list/tuple/dict of tensors.
    """
    ...

def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple of tensors)."
    ...

def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    ...

def nested_xla_mesh_reduce(tensors, name):
    ...

def distributed_concat(tensor: torch.Tensor, num_total_examples: Optional[int] = ...) -> torch.Tensor:
    ...

def distributed_broadcast_scalars(scalars: List[Union[int, float]], num_total_examples: Optional[int] = ...) -> torch.Tensor:
    ...

def reissue_pt_warnings(caught_warnings):
    ...

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.

    Args:
        local_rank (:obj:`int`): The rank of the local process.
    """
    ...

class DistributedSamplerWithLoop(DistributedSampler):
    """
    Like a :obj:torch.utils.data.distributed.DistributedSampler` but loops at the end back to the beginning of the
    shuffled samples to make each process have a round multiple of batch_size samples.

    Args:
        dataset (:obj:`torch.utils.data.Dataset`):
            Dataset used for sampling.
        batch_size (:obj:`int`):
            The batch size used with this sampler
        kwargs:
            All other keyword arguments passed to :obj:`DistributedSampler`.
    """
    def __init__(self, dataset, batch_size, **kwargs) -> None:
        ...
    
    def __iter__(self):
        ...
    


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indices sequentially, making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training), which means that the model params won't
    have to be synced (i.e. will not hang for synchronization even if varied number of forward passes), we still add
    extra samples to the sampler to make it evenly divisible (like in `DistributedSampler`) to make it easy to `gather`
    or `reduce` resulting tensors at the end of the loop.
    """
    def __init__(self, dataset, num_replicas=..., rank=..., batch_size=...) -> None:
        ...
    
    def __iter__(self):
        ...
    
    def __len__(self):
        ...
    


def get_tpu_sampler(dataset: torch.utils.data.dataset.Dataset, bach_size: int):
    ...

def nested_new_like(arrays, num_samples, padding_index=...):
    """Create the same nested structure as `arrays` with a first dimension always at `num_samples`."""
    ...

def expand_like(arrays, new_seq_length, padding_index=...):
    """Expand the `arrays` so that the second dimension grows to `new_seq_length`. Uses `padding_index` for padding."""
    ...

def nested_truncate(tensors, limit):
    "Truncate `tensors` at `limit` (even if it's a nested list/tuple of tensors)."
    ...

class DistributedTensorGatherer:
    """
    A class responsible for properly gathering tensors (or nested list/tuple of tensors) on the CPU by chunks.

    If our dataset has 16 samples with a batch size of 2 on 3 processes and we gather then transfer on CPU at every
    step, our sampler will generate the following indices:

        :obj:`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1]`

    to get something of size a multiple of 3 (so that each process gets the same dataset length). Then process 0, 1 and
    2 will be responsible of making predictions for the following samples:

        - P0: :obj:`[0, 1, 2, 3, 4, 5]`
        - P1: :obj:`[6, 7, 8, 9, 10, 11]`
        - P2: :obj:`[12, 13, 14, 15, 0, 1]`

    The first batch treated on each process will be

        - P0: :obj:`[0, 1]`
        - P1: :obj:`[6, 7]`
        - P2: :obj:`[12, 13]`

    So if we gather at the end of the first batch, we will get a tensor (nested list/tuple of tensor) corresponding to
    the following indices:

        :obj:`[0, 1, 6, 7, 12, 13]`

    If we directly concatenate our results without taking any precautions, the user will then get the predictions for
    the indices in this order at the end of the prediction loop:

        :obj:`[0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1]`

    For some reason, that's not going to roll their boat. This class is there to solve that problem.

    Args:

        world_size (:obj:`int`):
            The number of processes used in the distributed training.
        num_samples (:obj:`int`):
            The number of samples in our dataset.
        make_multiple_of (:obj:`int`, `optional`):
            If passed, the class assumes the datasets passed to each process are made to be a multiple of this argument
            (by adding samples).
        padding_index (:obj:`int`, `optional`, defaults to -100):
            The padding index to use if the arrays don't all have the same sequence length.
    """
    def __init__(self, world_size, num_samples, make_multiple_of=..., padding_index=...) -> None:
        ...
    
    def add_arrays(self, arrays):
        """
        Add :obj:`arrays` to the internal storage, Will initialize the storage to the full size at the first arrays
        passed so that if we're bound to get an OOM, it happens at the beginning.
        """
        ...
    
    def finalize(self):
        """
        Return the properly gathered arrays and truncate to the number of samples (since the sampler added some extras
        to get each process a dataset of the same length).
        """
        ...
    


@dataclass
class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (:obj:`float`, `optional`, defaults to 0.1):
            The label smoothing factor.
        ignore_index (:obj:`int`, `optional`, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """
    epsilon: float = ...
    ignore_index: int = ...
    def __call__(self, model_output, labels):
        ...
    


def get_length_grouped_indices(lengths, batch_size, mega_batch_mult=..., generator=...):
    """
    Return a list of indices so that each slice of :obj:`batch_size` consecutive indices correspond to elements of
    similar lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size :obj:`mega_batch_mult * batch_size`
    - sorted by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of :obj:`batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """
    ...

class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """
    def __init__(self, dataset: Dataset, batch_size: int, lengths: Optional[List[int]] = ..., model_input_name: Optional[str] = ..., generator=...) -> None:
        ...
    
    def __len__(self):
        ...
    
    def __iter__(self):
        ...
    


class DistributedLengthGroupedSampler(DistributedSampler):
    r"""
    Distributed Sampler that samples indices in a way that groups together features of the dataset of roughly the same
    length while keeping a bit of randomness.
    """
    def __init__(self, dataset: Dataset, batch_size: int, num_replicas: Optional[int] = ..., rank: Optional[int] = ..., seed: int = ..., drop_last: bool = ..., lengths: Optional[List[int]] = ..., model_input_name: Optional[str] = ...) -> None:
        ...
    
    def __iter__(self) -> Iterator:
        ...
    


class ShardSampler(Sampler):
    """
    Sampler that shards batches between several processes. Dispatches indices batch by batch: on 2 processes with batch
    size 4, the first two batches are :obj:`[0, 1, 2, 3, 4, 5, 6, 7]` and :obj:`[8, 9, 10, 11, 12, 13, 14, 15]`, which
    shard into :obj:`[0, 1, 2, 3]` and :obj:`[8, 9, 10, 11]` for GPU-0 and :obj:`[4, 5, 6, 7]` and :obj:`[12, 13, 14,
    15]` for GPU-1.

    The sampler thus yields :obj:`[0, 1, 2, 3, 8, 9, 10, 11]` on GPU-0 and :obj:`[4, 5, 6, 7, 12, 13, 14, 15]` on
    GPU-1.
    """
    def __init__(self, dataset: Dataset, batch_size: int = ..., drop_last: bool = ..., num_processes: int = ..., process_index: int = ...) -> None:
        ...
    
    def __iter__(self):
        ...
    
    def __len__(self):
        ...
    


class IterableDatasetShard(IterableDataset):
    """
    Wraps a PyTorch :obj:`IterableDataset` to generate samples for one of the processes only. Instances of this class
    will always yield a number of samples that is a round multiple of the actual batch size (which is :obj:`batch_size
    x num_processes`). Depending on the value of the :obj:`drop_last` attribute, it will either stop the iteration at
    the first batch that would be too small or loop with indices from the beginning.

    On two processes with an iterable dataset yielding of :obj:`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]` with a batch
    size of 2:

    - the shard on process 0 will yield :obj:`[0, 1, 4, 5, 8, 9]` so will see batches :obj:`[0, 1]`, :obj:`[4, 5]`,
      :obj:`[8, 9]`
    - the shard on process 1 will yield :obj:`[2, 3, 6, 7, 10, 11]` so will see batches :obj:`[2, 3]`, :obj:`[6, 7]`,
      :obj:`[10, 11]`

    .. warning:

        If your IterableDataset implements some randomization that needs to be applied the same way on all processes
        (for instance, a shuffling), you should use a :obj:`torch.Generator` in a :obj:`generator` attribute of the
        :obj:`dataset` to generate your random numbers and call the
        :meth:`~transformers.trainer_pt_utils.IterableDatasetShard.set_epoch` method of this object. It will set the
        seed of this :obj:`generator` to :obj:`seed + epoch` on all processes before starting the iteration.
        Alternatively, you can also implement a :obj:`set_epoch()` method in your iterable dataset to deal with this.


    Args:
        dataset (:obj:`torch.utils.data.dataset.IterableDataset`):
            The batch sampler to split in several shards.
        batch_size (:obj:`int`, `optional`, defaults to 1):
            The size of the batches per shard.
        drop_last (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to drop the last incomplete batch or complete the last batches by using the samples from the
            beginning.
        num_processes (:obj:`int`, `optional`, defaults to 1):
            The number of processes running concurrently.
        process_index (:obj:`int`, `optional`, defaults to 0):
            The index of the current process.
        seed (:obj:`int`, `optional`, defaults to 0):
            A random seed that will be used for the random number generation in
            :meth:`~transformers.trainer_pt_utils.IterableDatasetShard.set_epoch`.
    """
    def __init__(self, dataset: IterableDataset, batch_size: int = ..., drop_last: bool = ..., num_processes: int = ..., process_index: int = ..., seed: int = ...) -> None:
        ...
    
    def set_epoch(self, epoch):
        ...
    
    def __iter__(self):
        ...
    


def metrics_format(self, metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Reformat Trainer metrics values to a human-readable format

    Args:
        metrics (:obj:`Dict[str, float]`):
            The metrics returned from train/evaluate/predict

    Returns:
        metrics (:obj:`Dict[str, float]`): The reformatted metrics
    """
    ...

def log_metrics(self, split, metrics):
    """
    Log metrics in a specially formatted way

    Under distributed environment this is done only for a process with rank 0.

    Args:
        split (:obj:`str`):
            Mode/split name: one of ``train``, ``eval``, ``test``
        metrics (:obj:`Dict[str, float]`):
            The metrics returned from train/evaluate/predictmetrics: metrics dict

    Notes on memory reports:

    In order to get memory usage report you need to install ``psutil``. You can do that with ``pip install psutil``.

    Now when this method is run, you will see a report that will include: ::

        init_mem_cpu_alloc_delta   =     1301MB
        init_mem_cpu_peaked_delta  =      154MB
        init_mem_gpu_alloc_delta   =      230MB
        init_mem_gpu_peaked_delta  =        0MB
        train_mem_cpu_alloc_delta  =     1345MB
        train_mem_cpu_peaked_delta =        0MB
        train_mem_gpu_alloc_delta  =      693MB
        train_mem_gpu_peaked_delta =        7MB

    **Understanding the reports:**

    - the first segment, e.g., ``train__``, tells you which stage the metrics are for. Reports starting with ``init_``
      will be added to the first stage that gets run. So that if only evaluation is run, the memory usage for the
      ``__init__`` will be reported along with the ``eval_`` metrics.
    - the third segment, is either ``cpu`` or ``gpu``, tells you whether it's the general RAM or the gpu0 memory
      metric.
    - ``*_alloc_delta`` - is the difference in the used/allocated memory counter between the end and the start of the
      stage - it can be negative if a function released more memory than it allocated.
    - ``*_peaked_delta`` - is any extra memory that was consumed and then freed - relative to the current allocated
      memory counter - it is never negative. When you look at the metrics of any stage you add up ``alloc_delta`` +
      ``peaked_delta`` and you know how much memory was needed to complete that stage.

    The reporting happens only for process of rank 0 and gpu 0 (if there is a gpu). Typically this is enough since the
    main process does the bulk of work, but it could be not quite so if model parallel is used and then other GPUs may
    use a different amount of gpu memory. This is also not the same under DataParallel where gpu0 may require much more
    memory than the rest since it stores the gradient and optimizer states for all participating GPUS. Perhaps in the
    future these reports will evolve to measure those too.

    The CPU RAM metric measures RSS (Resident Set Size) includes both the memory which is unique to the process and the
    memory shared with other processes. It is important to note that it does not include swapped out memory, so the
    reports could be imprecise.

    The CPU peak memory is measured using a sampling thread. Due to python's GIL it may miss some of the peak memory if
    that thread didn't get a chance to run when the highest memory was used. Therefore this report can be less than
    reality. Using ``tracemalloc`` would have reported the exact peak memory, but it doesn't report memory allocations
    outside of python. So if some C++ CUDA extension allocated its own memory it won't be reported. And therefore it
    was dropped in favor of the memory sampling approach, which reads the current process memory usage.

    The GPU allocated and peak memory reporting is done with ``torch.cuda.memory_allocated()`` and
    ``torch.cuda.max_memory_allocated()``. This metric reports only "deltas" for pytorch-specific allocations, as
    ``torch.cuda`` memory management system doesn't track any memory allocated outside of pytorch. For example, the
    very first cuda call typically loads CUDA kernels, which may take from 0.5 to 2GB of GPU memory.

    Note that this tracker doesn't account for memory allocations outside of :class:`~transformers.Trainer`'s
    ``__init__``, ``train``, ``evaluate`` and ``predict`` calls.

    Because ``evaluation`` calls may happen during ``train``, we can't handle nested invocations because
    ``torch.cuda.max_memory_allocated`` is a single counter, so if it gets reset by a nested eval call, ``train``'s
    tracker will report incorrect info. If this `pytorch issue <https://github.com/pytorch/pytorch/issues/16266>`__
    gets resolved it will be possible to change this class to be re-entrant. Until then we will only track the outer
    level of ``train``, ``evaluate`` and ``predict`` methods. Which means that if ``eval`` is called during ``train``,
    it's the latter that will account for its memory usage and that of the former.

    This also means that if any other tool that is used along the :class:`~transformers.Trainer` calls
    ``torch.cuda.reset_peak_memory_stats``, the gpu peak memory stats could be invalid. And the
    :class:`~transformers.Trainer` will disrupt the normal behavior of any such tools that rely on calling
    ``torch.cuda.reset_peak_memory_stats`` themselves.

    For best performance you may want to consider turning the memory profiling off for production runs.
    """
    ...

def save_metrics(self, split, metrics, combined=...):
    """
    Save metrics into a json file for that split, e.g. ``train_results.json``.

    Under distributed environment this is done only for a process with rank 0.

    Args:
        split (:obj:`str`):
            Mode/split name: one of ``train``, ``eval``, ``test``, ``all``
        metrics (:obj:`Dict[str, float]`):
            The metrics returned from train/evaluate/predict
        combined (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Creates combined metrics by updating ``all_results.json`` with metrics of this call

    To understand the metrics please read the docstring of :meth:`~transformers.Trainer.log_metrics`. The only
    difference is that raw unformatted numbers are saved in the current method.

    """
    ...

def save_state(self):
    """
    Saves the Trainer state, since Trainer.save_model saves only the tokenizer with the model

    Under distributed environment this is done only for a process with rank 0.
    """
    ...

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    ...

if is_sagemaker_mp_enabled():
    @smp.step()
    def smp_forward_backward(model, inputs, gradient_accumulation_steps=..., scaler=...):
        ...
    
    @smp.step()
    def smp_forward_only(model, inputs):
        ...
    
    def smp_gather(tensor):
        ...
    
    def smp_nested_concat(tensor):
        ...
    
