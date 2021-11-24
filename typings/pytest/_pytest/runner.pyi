
from typing import TYPE_CHECKING, Callable, Generic, List, Optional, Tuple, Type, Union

import attr
from _pytest.compat import final
from _pytest.config.argparsing import Parser
from _pytest.main import Session
from _pytest.nodes import Collector, Item
from _pytest.terminal import TerminalReporter
from typing_extensions import Literal

from .reports import BaseReport, CollectReport, TestReport

"""Basic collect and runtest protocol implementations."""
if TYPE_CHECKING:
    ...
def pytest_addoption(parser: Parser) -> None:
    ...

def pytest_terminal_summary(terminalreporter: TerminalReporter) -> None:
    ...

def pytest_sessionstart(session: Session) -> None:
    ...

def pytest_sessionfinish(session: Session) -> None:
    ...

def pytest_runtest_protocol(item: Item, nextitem: Optional[Item]) -> bool:
    ...

def runtestprotocol(item: Item, log: bool = ..., nextitem: Optional[Item] = ...) -> List[TestReport]:
    ...

def show_test_item(item: Item) -> None:
    """Show test function, parameters and the fixtures of the test item."""
    ...

def pytest_runtest_setup(item: Item) -> None:
    ...

def pytest_runtest_call(item: Item) -> None:
    ...

def pytest_runtest_teardown(item: Item, nextitem: Optional[Item]) -> None:
    ...

def pytest_report_teststatus(report: BaseReport) -> Optional[Tuple[str, str, str]]:
    ...

def call_and_report(item: Item, when: Literal[setup, call, teardown], log: bool = ..., **kwds) -> TestReport:
    ...

def check_interactive_exception(call: CallInfo[object], report: BaseReport) -> bool:
    """Check whether the call raised an exception that should be reported as
    interactive."""
    ...

def call_runtest_hook(item: Item, when: Literal[setup, call, teardown], **kwds) -> CallInfo[None]:
    ...

TResult = ...
@final
@attr.s(repr=False)
class CallInfo(Generic[TResult]):
    """Result/Exception info a function invocation.

    :param T result:
        The return value of the call, if it didn't raise. Can only be
        accessed if excinfo is None.
    :param Optional[ExceptionInfo] excinfo:
        The captured exception of the call, if it raised.
    :param float start:
        The system time when the call started, in seconds since the epoch.
    :param float stop:
        The system time when the call ended, in seconds since the epoch.
    :param float duration:
        The call duration, in seconds.
    :param str when:
        The context of invocation: "setup", "call", "teardown", ...
    """
    _result = ...
    excinfo = ...
    start = ...
    stop = ...
    duration = ...
    when = ...
    @property
    def result(self) -> TResult:
        ...
    
    @classmethod
    def from_call(cls, func: Callable[[], TResult], when: Literal[collect, setup, call, teardown], reraise: Optional[Union[Type[BaseException], Tuple[Type[BaseException], ...]]] = ...) -> CallInfo[TResult]:
        ...
    
    def __repr__(self) -> str:
        ...
    


def pytest_runtest_makereport(item: Item, call: CallInfo[None]) -> TestReport:
    ...

def pytest_make_collect_report(collector: Collector) -> CollectReport:
    ...

class SetupState:
    """Shared state for setting up/tearing down test items or collectors."""
    def __init__(self) -> None:
        ...
    
    def addfinalizer(self, finalizer: Callable[[], object], colitem) -> None:
        """Attach a finalizer to the given colitem."""
        ...
    
    def teardown_all(self) -> None:
        ...
    
    def teardown_exact(self, item, nextitem) -> None:
        ...
    
    def prepare(self, colitem) -> None:
        """Setup objects along the collector chain to the test-method."""
        ...
    


def collect_one_node(collector: Collector) -> CollectReport:
    ...

