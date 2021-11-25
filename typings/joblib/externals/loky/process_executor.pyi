"""
This type stub file was generated by pyright.
"""

import sys
import threading

from . import _base
from .backend.queues import Queue

"""Implements ProcessPoolExecutor.

The follow diagram and text describe the data-flow through the system:

|======================= In-process =====================|== Out-of-process ==|

+----------+     +----------+       +--------+     +-----------+    +---------+
|          |  => | Work Ids |       |        |     | Call Q    |    | Process |
|          |     +----------+       |        |     +-----------+    |  Pool   |
|          |     | ...      |       |        |     | ...       |    +---------+
|          |     | 6        |    => |        |  => | 5, call() | => |         |
|          |     | 7        |       |        |     | ...       |    |         |
| Process  |     | ...      |       | Local  |     +-----------+    | Process |
|  Pool    |     +----------+       | Worker |                      |  #1..n  |
| Executor |                        | Thread |                      |         |
|          |     +----------- +     |        |     +-----------+    |         |
|          | <=> | Work Items | <=> |        | <=  | Result Q  | <= |         |
|          |     +------------+     |        |     +-----------+    |         |
|          |     | 6: call()  |     |        |     | ...       |    |         |
|          |     |    future  |     +--------+     | 4, result |    |         |
|          |     | ...        |                    | 3, except |    |         |
+----------+     +------------+                    +-----------+    +---------+

Executor.submit() called:
- creates a uniquely numbered _WorkItem and adds it to the "Work Items" dict
- adds the id of the _WorkItem to the "Work Ids" queue

Local worker thread:
- reads work ids from the "Work Ids" queue and looks up the corresponding
  WorkItem from the "Work Items" dict: if the work item has been cancelled then
  it is simply removed from the dict, otherwise it is repackaged as a
  _CallItem and put in the "Call Q". New _CallItems are put in the "Call Q"
  until "Call Q" is full. NOTE: the size of the "Call Q" is kept small because
  calls placed in the "Call Q" can no longer be cancelled with Future.cancel().
- reads _ResultItems from "Result Q", updates the future stored in the
  "Work Items" dict and deletes the dict entry

Process #1..n:
- reads _CallItems from "Call Q", executes the calls, and puts the resulting
  _ResultItems in "Result Q"
"""
__author__ = ...
if sys.version_info[0] == 2:
    ...
MAX_DEPTH = ...
_CURRENT_DEPTH = ...
_MEMORY_LEAK_CHECK_DELAY = ...
_MAX_MEMORY_LEAK_SIZE = ...
class _ThreadWakeup:
    def __init__(self) -> None:
        ...
    
    def close(self): # -> None:
        ...
    
    def wakeup(self): # -> None:
        ...
    
    def clear(self): # -> None:
        ...
    


class _ExecutorFlags:
    """necessary references to maintain executor states without preventing gc

    It permits to keep the information needed by executor_manager_thread
    and crash_detection_thread to maintain the pool without preventing the
    garbage collection of unreferenced executors.
    """
    def __init__(self, shutdown_lock) -> None:
        ...
    
    def flag_as_shutting_down(self, kill_workers=...): # -> None:
        ...
    
    def flag_as_broken(self, broken): # -> None:
        ...
    


_threads_wakeups = ...
_global_shutdown = ...
process_pool_executor_at_exit = ...
EXTRA_QUEUED_CALLS = ...
class _RemoteTraceback(Exception):
    """Embed stringification of remote traceback in local traceback
    """
    def __init__(self, tb=...) -> None:
        ...
    
    def __str__(self) -> str:
        ...
    


class _ExceptionWithTraceback(BaseException):
    def __init__(self, exc) -> None:
        ...
    
    def __reduce__(self): # -> tuple[(exc: Unknown, tb: Unknown) -> Unknown, tuple[Unknown, str]]:
        ...
    


class _WorkItem:
    __slots__ = ...
    def __init__(self, future, fn, args, kwargs) -> None:
        ...
    


class _ResultItem:
    def __init__(self, work_id, exception=..., result=...) -> None:
        ...
    


class _CallItem:
    def __init__(self, work_id, fn, args, kwargs) -> None:
        ...
    
    def __call__(self):
        ...
    
    def __repr__(self): # -> str:
        ...
    


class _SafeQueue(Queue):
    """Safe Queue set exception to the future object linked to a job"""
    def __init__(self, max_size=..., ctx=..., pending_work_items=..., running_work_items=..., thread_wakeup=..., reducers=...) -> None:
        ...
    


class _ExecutorManagerThread(threading.Thread):
    """Manages the communication between this process and the worker processes.

    The manager is run in a local thread.

    Args:
        executor: A reference to the ProcessPoolExecutor that owns
            this thread. A weakref will be own by the manager as well as
            references to internal objects used to introspect the state of
            the executor.
    """
    def __init__(self, executor) -> None:
        ...
    
    def run(self): # -> None:
        ...
    
    def add_call_item_to_queue(self): # -> None:
        ...
    
    def wait_result_broken_or_wakeup(self): # -> tuple[_RemoteTraceback | Unknown | None, bool, BrokenProcessPool | TerminatedWorkerError | None]:
        ...
    
    def process_result_item(self, result_item):
        ...
    
    def is_shutting_down(self): # -> bool:
        ...
    
    def terminate_broken(self, bpe): # -> None:
        ...
    
    def flag_executor_shutting_down(self): # -> None:
        ...
    
    def kill_workers(self, reason=...):
        ...
    
    def shutdown_workers(self): # -> None:
        ...
    
    def join_executor_internals(self): # -> None:
        ...
    
    def get_n_children_alive(self): # -> int:
        ...
    


_system_limits_checked = ...
_system_limited = ...
class LokyRecursionError(RuntimeError):
    """Raised when a process try to spawn too many levels of nested processes.
    """
    ...


class BrokenProcessPool(_BPPException):
    """
    Raised when the executor is broken while a future was in the running state.
    The cause can an error raised when unpickling the task in the worker
    process or when unpickling the result value in the parent process. It can
    also be caused by a worker process being terminated unexpectedly.
    """
    ...


class TerminatedWorkerError(BrokenProcessPool):
    """
    Raised when a process in a ProcessPoolExecutor terminated abruptly
    while a future was in the running state.
    """
    ...


BrokenExecutor = BrokenProcessPool
class ShutdownExecutorError(RuntimeError):
    """
    Raised when a ProcessPoolExecutor is shutdown while a future was in the
    running or pending state.
    """
    ...


class ProcessPoolExecutor(_base.Executor):
    _at_exit = ...
    def __init__(self, max_workers=..., job_reducers=..., result_reducers=..., timeout=..., context=..., initializer=..., initargs=..., env=...) -> None:
        """Initializes a new ProcessPoolExecutor instance.

        Args:
            max_workers: int, optional (default: cpu_count())
                The maximum number of processes that can be used to execute the
                given calls. If None or not given then as many worker processes
                will be created as the number of CPUs the current process
                can use.
            job_reducers, result_reducers: dict(type: reducer_func)
                Custom reducer for pickling the jobs and the results from the
                Executor. If only `job_reducers` is provided, `result_reducer`
                will use the same reducers
            timeout: int, optional (default: None)
                Idle workers exit after timeout seconds. If a new job is
                submitted after the timeout, the executor will start enough
                new Python processes to make sure the pool of workers is full.
            context: A multiprocessing context to launch the workers. This
                object should provide SimpleQueue, Queue and Process.
            initializer: An callable used to initialize worker processes.
            initargs: A tuple of arguments to pass to the initializer.
            env: A dict of environment variable to overwrite in the child
                process. The environment variables are set before any module is
                loaded. Note that this only works with the loky context and it
                is unreliable under windows with Python < 3.6.
        """
        ...
    
    def submit(self, fn, *args, **kwargs): # -> Future:
        ...
    
    def map(self, fn, *iterables, **kwargs): # -> Generator[Unknown, None, None]:
        """Returns an iterator equivalent to map(fn, iter).

        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            chunksize: If greater than one, the iterables will be chopped into
                chunks of size chunksize and submitted to the process pool.
                If set to one, the items in the list will be sent one at a
                time.

        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.

        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """
        ...
    
    def shutdown(self, wait=..., kill_workers=...): # -> None:
        ...
    


