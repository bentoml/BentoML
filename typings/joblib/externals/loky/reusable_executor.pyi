import threading
from .process_executor import ProcessPoolExecutor

__all__ = ["get_reusable_executor"]
STRING_TYPE = ...
_executor_lock = threading.RLock()
_next_executor_id = ...
_executor = ...
_executor_kwargs = ...

def get_reusable_executor(
    max_workers=...,
    context=...,
    timeout=...,
    kill_workers=...,
    reuse=...,
    job_reducers=...,
    result_reducers=...,
    initializer=...,
    initargs=...,
    env=...,
): ...

class _ReusablePoolExecutor(ProcessPoolExecutor):
    def __init__(
        self,
        submit_resize_lock,
        max_workers=...,
        context=...,
        timeout=...,
        executor_id=...,
        job_reducers=...,
        result_reducers=...,
        initializer=...,
        initargs=...,
        env=...,
    ) -> None: ...
    @classmethod
    def get_reusable_executor(
        cls,
        max_workers=...,
        context=...,
        timeout=...,
        kill_workers=...,
        reuse=...,
        job_reducers=...,
        result_reducers=...,
        initializer=...,
        initargs=...,
        env=...,
    ): ...
