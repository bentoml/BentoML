from .externals.loky.reusable_executor import _ReusablePoolExecutor

"""Utility function to construct a loky.ReusableExecutor with custom pickler.

This module provides efficient ways of working with data stored in
shared memory with numpy.memmap arrays without inducing any memory
copy between the parent and child processes.
"""
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
    ):
        """Factory for ReusableExecutor with automatic memmapping for large numpy
        arrays.
        """
        ...
    def terminate(self, kill_workers=...): ...

class _TestingMemmappingExecutor(MemmappingExecutor):
    """Wrapper around ReusableExecutor to ease memmapping testing with Pool
    and Executor. This is only for testing purposes.

    """

    def apply_async(self, func, args):  # -> Future:
        """Schedule a func to be run"""
        ...
    def map(self, f, *args): ...
