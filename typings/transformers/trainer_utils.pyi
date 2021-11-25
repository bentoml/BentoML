

from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import numpy as np

from .file_utils import ExplicitEnum, is_tf_available, is_torch_available

if is_torch_available():
    ...
if is_tf_available():
    ...
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    ...

class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray
    ...


class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]
    ...


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]
    ...


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float
    metrics: Dict[str, float]
    ...


PREFIX_CHECKPOINT_DIR = ...
_re_checkpoint = ...
def get_last_checkpoint(folder):
    ...

class IntervalStrategy(ExplicitEnum):
    NO = ...
    STEPS = ...
    EPOCH = ...


class EvaluationStrategy(ExplicitEnum):
    NO = ...
    STEPS = ...
    EPOCH = ...


class BestRun(NamedTuple):
    """
    The best run found by an hyperparameter search (see :class:`~transformers.Trainer.hyperparameter_search`).

    Parameters:
        run_id (:obj:`str`):
            The id of the best run (if models were saved, the corresponding checkpoint will be in the folder ending
            with run-{run_id}).
        objective (:obj:`float`):
            The objective that was obtained for this run.
        hyperparameters (:obj:`Dict[str, Any]`):
            The hyperparameters picked to get this run.
    """
    run_id: str
    objective: float
    hyperparameters: Dict[str, Any]
    ...


def default_compute_objective(metrics: Dict[str, float]) -> float:
    """
    The default objective to maximize/minimize when doing an hyperparameter search. It is the evaluation loss if no
    metrics are provided to the :class:`~transformers.Trainer`, the sum of all metrics otherwise.

    Args:
        metrics (:obj:`Dict[str, float]`): The metrics returned by the evaluate method.

    Return:
        :obj:`float`: The objective to minimize or maximize
    """
    ...

def default_hp_space_optuna(trial) -> Dict[str, float]:
    ...

def default_hp_space_ray(trial) -> Dict[str, float]:
    ...

class HPSearchBackend(ExplicitEnum):
    OPTUNA = ...
    RAY = ...


default_hp_space = ...
def is_main_process(local_rank):
    """
    Whether or not the current process is the local process, based on `xm.get_ordinal()` (for TPUs) first, then on
    `local_rank`.
    """
    ...

def total_processes_number(local_rank):
    """
    Return the number of processes launched in parallel. Works with `torch.distributed` and TPUs.
    """
    ...

def speed_metrics(split, start_time, num_samples=..., num_steps=...):
    """
    Measure and return speed performance metrics.

    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:

    - split: name to prefix metric (like train, eval, test...)
    - start_time: operation start time
    - num_samples: number of samples processed
    """
    ...

class SchedulerType(ExplicitEnum):
    LINEAR = ...
    COSINE = ...
    COSINE_WITH_RESTARTS = ...
    POLYNOMIAL = ...
    CONSTANT = ...
    CONSTANT_WITH_WARMUP = ...


class TrainerMemoryTracker:
    """
    A helper class that tracks cpu and gpu memory.

    This class will silently skip unless ``psutil`` is available. Install with ``pip install psutil``.

    When a stage completes, it can pass metrics dict to update with the memory metrics gathered during this stage.

    Example ::

        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()
        code ...
        metrics = {"train_runtime": 10.5}
        self._memory_tracker.stop_and_update_metrics(metrics)

    At the moment GPU tracking is only for ``pytorch``, but can be extended to support ``tensorflow``.

    To understand this class' intricacies please read the documentation of :meth:`~transformers.Trainer.log_metrics`.

    """
    stages = ...
    def __init__(self, skip_memory_metrics=...) -> None:
        ...
    
    def derive_stage(self):
        """derives the stage/caller name automatically"""
        ...
    
    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        ...
    
    def peak_monitor_func(self):
        ...
    
    def start(self):
        """start tracking for the caller's stage"""
        ...
    
    def stop(self, stage):
        """stop tracking for the passed stage"""
        ...
    
    def update_metrics(self, stage, metrics):
        """updates the metrics"""
        ...
    
    def stop_and_update_metrics(self, metrics=...):
        """combine stop and metrics update in one call for simpler code"""
        ...
    


def denumpify_detensorize(metrics):
    """
    Recursively calls `.item()` on the element of the dictionary passed
    """
    ...

def number_of_arguments(func):
    """
    Return the number of arguments of the passed function, even if it's a partial function.
    """
    ...

class ShardedDDPOption(ExplicitEnum):
    SIMPLE = ...
    ZERO_DP_2 = ...
    ZERO_DP_3 = ...
    OFFLOAD = ...
    AUTO_WRAP = ...


