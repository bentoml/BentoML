

import os
from abc import ABC
from typing import Callable, Dict, List, Optional, Tuple, Union

"""Training Library containing training routines."""
def print_evaluation(period=..., show_stdv=...): # -> (env: Unknown) -> None:
    """Create a callback that print evaluation result.

    We print the evaluation results every **period** iterations
    and on the first and the last iterations.

    Parameters
    ----------
    period : int
        The period to log the evaluation results

    show_stdv : bool, optional
         Whether show stdv if provided

    Returns
    -------
    callback : function
        A callback that print evaluation every period iterations.
    """
    ...

def record_evaluation(eval_result): # -> (env: Unknown) -> None:
    """Create a call back that records the evaluation history into **eval_result**.

    Parameters
    ----------
    eval_result : dict
       A dictionary to store the evaluation results.

    Returns
    -------
    callback : function
        The requested callback function.
    """
    ...

def reset_learning_rate(learning_rates): # -> (env: Unknown) -> None:
    """Reset learning rate after iteration 1

    NOTE: the initial learning rate will still take in-effect on first iteration.

    Parameters
    ----------
    learning_rates: list or function
        List of learning rate for each boosting round
        or a customized function that calculates eta in terms of
        current number of round and the total number of boosting round (e.g.
        yields learning rate decay)

        * list ``l``: ``eta = l[boosting_round]``
        * function ``f``: ``eta = f(boosting_round, num_boost_round)``

    Returns
    -------
    callback : function
        The requested callback function.
    """
    ...

def early_stop(stopping_rounds, maximize=..., verbose=...): # -> (env: Unknown) -> None:
    """Create a callback that activates early stoppping.

    Validation error needs to decrease at least
    every **stopping_rounds** round(s) to continue training.
    Requires at least one item in **evals**.
    If there's more than one, will use the last.
    Returns the model from the last iteration (not the best one).
    If early stopping occurs, the model will have three additional fields:
    ``bst.best_score``, ``bst.best_iteration`` and ``bst.best_ntree_limit``.
    (Use ``bst.best_ntree_limit`` to get the correct value if ``num_parallel_tree``
    and/or ``num_class`` appears in the parameters)

    Parameters
    ----------
    stopping_rounds : int
       The stopping rounds before the trend occur.

    maximize : bool
        Whether to maximize evaluation metric.

    verbose : optional, bool
        Whether to print message about early stopping information.

    Returns
    -------
    callback : function
        The requested callback function.
    """
    ...

class TrainingCallback(ABC):
    '''Interface for training callback.

    .. versionadded:: 1.3.0

    '''
    def __init__(self) -> None:
        ...
    
    def before_training(self, model):
        '''Run before training starts.'''
        ...
    
    def after_training(self, model):
        '''Run after training is finished.'''
        ...
    
    def before_iteration(self, model, epoch: int, evals_log: CallbackContainer.EvalsLog) -> bool:
        '''Run before each iteration.  Return True when training should stop.'''
        ...
    
    def after_iteration(self, model, epoch: int, evals_log: CallbackContainer.EvalsLog) -> bool:
        '''Run after each iteration.  Return True when training should stop.'''
        ...
    


class CallbackContainer:
    '''A special callback for invoking a list of other callbacks.

    .. versionadded:: 1.3.0

    '''
    EvalsLog = Dict[str, Dict[str, Union[List[float], List[Tuple[float, float]]]]]
    def __init__(self, callbacks: List[TrainingCallback], metric: Callable = ..., is_cv: bool = ...) -> None:
        ...
    
    def before_training(self, model): # -> Booster:
        '''Function called before training.'''
        ...
    
    def after_training(self, model): # -> Booster:
        '''Function called after training.'''
        ...
    
    def before_iteration(self, model, epoch, dtrain, evals) -> bool:
        '''Function called before training iteration.'''
        ...
    
    def after_iteration(self, model, epoch, dtrain, evals) -> bool:
        '''Function called after training iteration.'''
        ...
    


class LearningRateScheduler(TrainingCallback):
    '''Callback function for scheduling learning rate.

    .. versionadded:: 1.3.0

    Parameters
    ----------

    learning_rates : callable/collections.Sequence
        If it's a callable object, then it should accept an integer parameter
        `epoch` and returns the corresponding learning rate.  Otherwise it
        should be a sequence like list or tuple with the same size of boosting
        rounds.

    '''
    def __init__(self, learning_rates) -> None:
        ...
    
    def after_iteration(self, model, epoch, evals_log) -> bool:
        ...
    


class EarlyStopping(TrainingCallback):
    """Callback function for early stopping

    .. versionadded:: 1.3.0

    Parameters
    ----------
    rounds
        Early stopping rounds.
    metric_name
        Name of metric that is used for early stopping.
    data_name
        Name of dataset that is used for early stopping.
    maximize
        Whether to maximize evaluation metric.  None means auto (discouraged).
    save_best
        Whether training should return the best model or the last model.
    """
    def __init__(self, rounds: int, metric_name: Optional[str] = ..., data_name: Optional[str] = ..., maximize: Optional[bool] = ..., save_best: Optional[bool] = ...) -> None:
        ...
    
    def before_training(self, model):
        ...
    
    def after_iteration(self, model, epoch: int, evals_log: CallbackContainer.EvalsLog) -> bool:
        ...
    
    def after_training(self, model):
        ...
    


class EvaluationMonitor(TrainingCallback):
    '''Print the evaluation result at each iteration.

    .. versionadded:: 1.3.0

    Parameters
    ----------

    metric : callable
        Extra user defined metric.
    rank : int
        Which worker should be used for printing the result.
    period : int
        How many epoches between printing.
    show_stdv : bool
        Used in cv to show standard deviation.  Users should not specify it.
    '''
    def __init__(self, rank=..., period=..., show_stdv=...) -> None:
        ...
    
    def after_iteration(self, model, epoch: int, evals_log: CallbackContainer.EvalsLog) -> bool:
        ...
    
    def after_training(self, model):
        ...
    


class TrainingCheckPoint(TrainingCallback):
    '''Checkpointing operation.

    .. versionadded:: 1.3.0

    Parameters
    ----------

    directory : os.PathLike
        Output model directory.
    name : str
        pattern of output model file.  Models will be saved as name_0.json, name_1.json,
        name_2.json ....
    as_pickle : boolean
        When set to Ture, all training parameters will be saved in pickle format, instead
        of saving only the model.
    iterations : int
        Interval of checkpointing.  Checkpointing is slow so setting a larger number can
        reduce performance hit.

    '''
    def __init__(self, directory: os.PathLike, name: str = ..., as_pickle=..., iterations: int = ...) -> None:
        ...
    
    def after_iteration(self, model, epoch: int, evals_log: CallbackContainer.EvalsLog) -> bool:
        ...
    


class LegacyCallbacks:
    '''Adapter for legacy callback functions.

    .. versionadded:: 1.3.0

    Parameters
    ----------

    callbacks : Sequence
        A sequence of legacy callbacks (callbacks that are not instance of
        TrainingCallback)
    start_iteration : int
        Begining iteration.
    end_iteration : int
        End iteration, normally is the number of boosting rounds.
    evals : Sequence
        Sequence of evaluation dataset tuples.
    feval : Custom evaluation metric.
    '''
    def __init__(self, callbacks, start_iteration, end_iteration, feval, cvfolds=...) -> None:
        ...
    
    def before_training(self, model):
        '''Nothing to do for legacy callbacks'''
        ...
    
    def after_training(self, model):
        '''Nothing to do for legacy callbacks'''
        ...
    
    def before_iteration(self, model, epoch, dtrain, evals): # -> Literal[False]:
        '''Called before each iteration.'''
        ...
    
    def after_iteration(self, model, epoch, dtrain, evals):
        '''Called after each iteration.'''
        ...
    


