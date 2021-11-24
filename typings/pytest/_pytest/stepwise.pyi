
from typing import TYPE_CHECKING, List, Optional

import pytest
from _pytest import nodes
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.main import Session
from _pytest.reports import TestReport

if TYPE_CHECKING:
    ...
STEPWISE_CACHE_DIR = ...
def pytest_addoption(parser: Parser) -> None:
    ...

@pytest.hookimpl
def pytest_configure(config: Config) -> None:
    ...

def pytest_sessionfinish(session: Session) -> None:
    ...

class StepwisePlugin:
    def __init__(self, config: Config) -> None:
        ...
    
    def pytest_sessionstart(self, session: Session) -> None:
        ...
    
    def pytest_collection_modifyitems(self, config: Config, items: List[nodes.Item]) -> None:
        ...
    
    def pytest_runtest_logreport(self, report: TestReport) -> None:
        ...
    
    def pytest_report_collectionfinish(self) -> Optional[str]:
        ...
    
    def pytest_sessionfinish(self) -> None:
        ...
    


