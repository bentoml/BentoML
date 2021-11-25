from typing import Any, Callable, List, Optional

import numpy as np

"""Distributed XGBoost Rabit related API."""

def init(args: Optional[List[bytes]] = ...) -> None:
    """Initialize the rabit library with arguments"""
    ...

def finalize() -> None:
    """Finalize the process, notify tracker everything is done."""
    ...

def get_rank() -> int:
    """Get rank of current process.

    Returns
    -------
    rank : int
        Rank of current process.
    """
    ...

def get_world_size() -> int:
    """Get total number workers.

    Returns
    -------
    n : int
        Total number of process.
    """
    ...

def is_distributed() -> int:
    """If rabit is distributed."""
    ...

def tracker_print(msg: Any) -> None:
    """Print message to the tracker.

    This function can be used to communicate the information of
    the progress to the tracker

    Parameters
    ----------
    msg : str
        The message to be printed to tracker.
    """
    ...

def get_processor_name() -> bytes:
    """Get the processor name.

    Returns
    -------
    name : str
        the name of processor(host)
    """
    ...

T = ...

def broadcast(data: T, root: int) -> T:
    """Broadcast object from one node to all other nodes.

    Parameters
    ----------
    data : any type that can be pickled
        Input data, if current rank does not equal root, this can be None
    root : int
        Rank of the node to broadcast data from.

    Returns
    -------
    object : int
        the result of broadcast.
    """
    ...

DTYPE_ENUM__ = ...

class Op:
    """Supported operations for rabit."""

    MAX = ...
    MIN = ...
    SUM = ...
    OR = ...

def allreduce(
    data: np.ndarray, op: int, prepare_fun: Optional[Callable[[np.ndarray], None]] = ...
) -> np.ndarray:
    """Perform allreduce, return the result.

    Parameters
    ----------
    data: numpy array
        Input data.
    op: int
        Reduction operators, can be MIN, MAX, SUM, BITOR
    prepare_fun: function
        Lazy preprocessing function, if it is not None, prepare_fun(data)
        will be called by the function before performing allreduce, to initialize the data
        If the result of Allreduce can be recovered directly,
        then prepare_fun will NOT be called

    Returns
    -------
    result : array_like
        The result of allreduce, have same shape as data

    Notes
    -----
    This function is not thread-safe.
    """
    ...

def version_number() -> int:
    """Returns version number of current stored model.

    This means how many calls to CheckPoint we made so far.

    Returns
    -------
    version : int
        Version number of currently stored model
    """
    ...
