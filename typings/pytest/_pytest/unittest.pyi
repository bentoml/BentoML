
import types
import unittest
from typing import (
    TYPE_CHECKING,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from _pytest.config import hookimpl
from _pytest.nodes import Collector, Item
from _pytest.python import Class, Function, PyCollector
from _pytest.runner import CallInfo

"""Discover and run std-library "unittest" style tests."""
if TYPE_CHECKING:
    _SysExcInfoType = Union[Tuple[Type[BaseException], BaseException, types.TracebackType], Tuple[None, None, None]],
def pytest_pycollect_makeitem(collector: PyCollector, name: str, obj: object) -> Optional[UnitTestCase]:
    ...

class UnitTestCase(Class):
    nofuncargs = ...
    def collect(self) -> Iterable[Union[Item, Collector]]:
        ...
    


class TestCaseFunction(Function):
    nofuncargs = ...
    _excinfo: Optional[List[_pytest._code.ExceptionInfo[BaseException]]] = ...
    _testcase: Optional[unittest.TestCase] = ...
    def setup(self) -> None:
        ...
    
    def teardown(self) -> None:
        ...
    
    def startTest(self, testcase: unittest.TestCase) -> None:
        ...
    
    def addError(self, testcase: unittest.TestCase, rawexcinfo: _SysExcInfoType) -> None:
        ...
    
    def addFailure(self, testcase: unittest.TestCase, rawexcinfo: _SysExcInfoType) -> None:
        ...
    
    def addSkip(self, testcase: unittest.TestCase, reason: str) -> None:
        ...
    
    def addExpectedFailure(self, testcase: unittest.TestCase, rawexcinfo: _SysExcInfoType, reason: str = ...) -> None:
        ...
    
    def addUnexpectedSuccess(self, testcase: unittest.TestCase, reason: str = ...) -> None:
        ...
    
    def addSuccess(self, testcase: unittest.TestCase) -> None:
        ...
    
    def stopTest(self, testcase: unittest.TestCase) -> None:
        ...
    
    def runtest(self) -> None:
        ...
    


@hookimpl(tryfirst=True)
def pytest_runtest_makereport(item: Item, call: CallInfo[None]) -> None:
    ...

@hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: Item) -> Generator[None, None, None]:
    ...

def check_testcase_implements_trial_reporter(done: List[int] = ...) -> None:
    ...

