
import types
from typing import TYPE_CHECKING, Any, Callable, Generator, List, Optional, Tuple, Type

from _pytest._code import ExceptionInfo
from _pytest.config import Config, PytestPluginManager, hookimpl
from _pytest.config.argparsing import Parser
from _pytest.nodes import Node
from _pytest.reports import BaseReport
from _pytest.runner import CallInfo

"""Interactive debugging with PDB, the Python Debugger."""
if TYPE_CHECKING:
    ...
def pytest_addoption(parser: Parser) -> None:
    ...

def pytest_configure(config: Config) -> None:
    ...

class pytestPDB:
    """Pseudo PDB that defers to the real pdb."""
    _pluginmanager: Optional[PytestPluginManager] = ...
    _config: Optional[Config] = ...
    _saved: List[Tuple[Callable[..., None], Optional[PytestPluginManager], Optional[Config]]] = ...
    _recursive_debug = ...
    _wrapped_pdb_cls: Optional[Tuple[Type[Any], Type[Any]]] = ...
    @classmethod
    def set_trace(cls, *args, **kwargs) -> None:
        """Invoke debugging via ``Pdb.set_trace``, dropping any IO capturing."""
        ...
    


class PdbInvoke:
    def pytest_exception_interact(self, node: Node, call: CallInfo[Any], report: BaseReport) -> None:
        ...
    
    def pytest_internalerror(self, excinfo: ExceptionInfo[BaseException]) -> None:
        ...
    


class PdbTrace:
    @hookimpl(hookwrapper=True)
    def pytest_pyfunc_call(self, pyfuncitem) -> Generator[None, None, None]:
        ...
    


def wrap_pytest_function_for_tracing(pyfuncitem): # -> None:
    """Change the Python function object of the given Function item by a
    wrapper which actually enters pdb before calling the python function
    itself, effectively leaving the user in the pdb prompt in the first
    statement of the function."""
    ...

def maybe_wrap_pytest_function_for_tracing(pyfuncitem): # -> None:
    """Wrap the given pytestfunct item for tracing support if --trace was given in
    the command line."""
    ...

def post_mortem(t: types.TracebackType) -> None:
    ...

