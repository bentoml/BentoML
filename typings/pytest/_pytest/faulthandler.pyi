
from typing import Generator

import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item

fault_handler_stderr_key = ...
def pytest_addoption(parser: Parser) -> None:
    ...

def pytest_configure(config: Config) -> None:
    ...

class FaultHandlerHooks:
    """Implements hooks that will actually install fault handler before tests execute,
    as well as correctly handle pdb and internal errors."""
    def pytest_configure(self, config: Config) -> None:
        ...
    
    def pytest_unconfigure(self, config: Config) -> None:
        ...
    
    @staticmethod
    def get_timeout_config_value(config): # -> float:
        ...
    
    @pytest.hookimpl(hookwrapper=True, trylast=True)
    def pytest_runtest_protocol(self, item: Item) -> Generator[None, None, None]:
        ...
    
    @pytest.hookimpl(tryfirst=True)
    def pytest_enter_pdb(self) -> None:
        """Cancel any traceback dumping due to timeout before entering pdb."""
        ...
    
    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(self) -> None:
        """Cancel any traceback dumping due to an interactive exception being
        raised."""
        ...
    


