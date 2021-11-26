from multiprocessing.pool import Pool
from pickle import Pickler

class CustomizablePickler(Pickler):
    def __init__(self, writer, reducers=..., protocol=...) -> None: ...
    def register(self, type, reduce_func): ...

class CustomizablePicklingQueue:
    def __init__(self, context, reducers=...) -> None: ...
    def __getstate__(self): ...
    def __setstate__(self, state): ...
    def empty(self): ...

class PicklingPool(Pool):
    def __init__(
        self, processes=..., forward_reducers=..., backward_reducers=..., **kwargs
    ) -> None: ...

class MemmappingPool(PicklingPool):
    def __init__(
        self,
        processes=...,
        temp_folder=...,
        max_nbytes=...,
        mmap_mode=...,
        forward_reducers=...,
        backward_reducers=...,
        verbose=...,
        context_id=...,
        prewarm=...,
        **kwargs
    ) -> None: ...
