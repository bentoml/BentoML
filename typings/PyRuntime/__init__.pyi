from importlib.machinery import ExtensionFileLoader, ModuleSpec
from typing import Any, Callable, List, Literal, overload
import numpy as np

__spec__: ModuleSpec
__loader__: ExtensionFileLoader
__name__: str = ...
__package__: str = ...
__doc__: str = ...
__file__: str = ...

class ExecutionSession:
    _entryPointFunc: Callable[..., Any]
    @overload
    def __init__(
        self, sharedLibPath: str, entryPointName: Literal["run_main_graph"]
    ) -> None: ...
    @overload
    def __init__(self, sharedLibPath: str, entryPointName: str) -> None: ...
    def run(
        self, input: np.ndarray[Any, np.dtype[Any]]
    ) -> List[np.ndarray[Any, np.dtype[Any]]]: ...
