from .externals.loky.reusable_executor import _ReusablePoolExecutor

_executor_args = ...

def get_memmapping_executor(n_jobs, **kwargs): ...

class MemmappingExecutor(_ReusablePoolExecutor):
    @classmethod
    def get_memmapping_executor(
        cls,
        n_jobs,
        timeout=...,
        initializer=...,
        initargs=...,
        env=...,
        temp_folder=...,
        context_id=...,
        **backend_args
    ): ...
    def terminate(self, kill_workers=...): ...

class _TestingMemmappingExecutor(MemmappingExecutor):
    def apply_async(self, func, args): ...
