
import doctest
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import pytest
from _pytest._code.code import ExceptionInfo, ReprFileLocation, TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.config.argparsing import Parser
from _pytest.nodes import Collector

"""Discover and run doctests in modules and test files."""
if TYPE_CHECKING:
    ...
DOCTEST_REPORT_CHOICE_NONE = ...
DOCTEST_REPORT_CHOICE_CDIFF = ...
DOCTEST_REPORT_CHOICE_NDIFF = ...
DOCTEST_REPORT_CHOICE_UDIFF = ...
DOCTEST_REPORT_CHOICE_ONLY_FIRST_FAILURE = ...
DOCTEST_REPORT_CHOICES = ...
RUNNER_CLASS = ...
CHECKER_CLASS: Optional[Type[doctest.OutputChecker]] = ...
def pytest_addoption(parser: Parser) -> None:
    ...

def pytest_unconfigure() -> None:
    ...

def pytest_collect_file(path: py.path.local, parent: Collector) -> Optional[Union[DoctestModule, DoctestTextfile]]:
    ...

class ReprFailDoctest(TerminalRepr):
    def __init__(self, reprlocation_lines: Sequence[Tuple[ReprFileLocation, Sequence[str]]]) -> None:
        ...
    
    def toterminal(self, tw: TerminalWriter) -> None:
        ...
    


class MultipleDoctestFailures(Exception):
    def __init__(self, failures: Sequence[doctest.DocTestFailure]) -> None:
        ...
    


class DoctestItem(pytest.Item):
    def __init__(self, name: str, parent: Union[DoctestTextfile, DoctestModule], runner: Optional[doctest.DocTestRunner] = ..., dtest: Optional[doctest.DocTest] = ...) -> None:
        ...
    
    @classmethod
    def from_parent(cls, parent: Union[DoctestTextfile, DoctestModule], *, name: str, runner: doctest.DocTestRunner, dtest: doctest.DocTest): # -> Any:
        """The public named constructor."""
        ...
    
    def setup(self) -> None:
        ...
    
    def runtest(self) -> None:
        ...
    
    def repr_failure(self, excinfo: ExceptionInfo[BaseException]) -> Union[str, TerminalRepr]:
        ...
    
    def reportinfo(self): # -> tuple[local | Any | None, int | None, str]:
        ...
    


def get_optionflags(parent): # -> int:
    ...

class DoctestTextfile(pytest.Module):
    obj = ...
    def collect(self) -> Iterable[DoctestItem]:
        ...
    


class DoctestModule(pytest.Module):
    def collect(self) -> Iterable[DoctestItem]:
        class MockAwareDocTestFinder(doctest.DocTestFinder):
            """A hackish doctest finder that overrides stdlib internals to fix a stdlib bug.

            https://github.com/pytest-dev/pytest/issues/3456
            https://bugs.python.org/issue25532
            """
            ...
        
        
    


@pytest.fixture(scope="session")
def doctest_namespace() -> Dict[str, Any]:
    """Fixture that returns a :py:class:`dict` that will be injected into the
    namespace of doctests."""
    ...

