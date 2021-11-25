import sys

from .compat import BaseProcess

class LokyProcess(BaseProcess):
    _start_method = ...
    def __init__(
        self,
        group=...,
        target=...,
        name=...,
        args=...,
        kwargs=...,
        daemon=...,
        init_main_module=...,
        env=...,
    ) -> None: ...
    if sys.version_info < (3, 3): ...
    if sys.version_info < (3, 4): ...

class LokyInitMainProcess(LokyProcess):
    _start_method = ...
    def __init__(
        self, group=..., target=..., name=..., args=..., kwargs=..., daemon=...
    ) -> None: ...

class AuthenticationKey(bytes):
    def __reduce__(self): ...
