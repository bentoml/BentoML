
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

from _pytest._code.code import ExceptionInfo, TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.compat import final
from _pytest.nodes import Collector, Item
from _pytest.runner import CallInfo
from typing_extensions import Literal

if TYPE_CHECKING:
    ...
def getworkerinfoline(node): # -> str:
    ...

_R = ...
class BaseReport:
    when: Optional[str]
    location: Optional[Tuple[str, Optional[int], str]]
    longrepr: Union[None, ExceptionInfo[BaseException], Tuple[str, int, str], str, TerminalRepr]
    sections: List[Tuple[str, str]]
    nodeid: str
    def __init__(self, **kw: Any) -> None:
        ...
    
    if TYPE_CHECKING:
        def __getattr__(self, key: str) -> Any:
            ...
        
    def toterminal(self, out: TerminalWriter) -> None:
        ...
    
    def get_sections(self, prefix: str) -> Iterator[Tuple[str, str]]:
        ...
    
    @property
    def longreprtext(self) -> str:
        """Read-only property that returns the full string representation of
        ``longrepr``.

        .. versionadded:: 3.0
        """
        ...
    
    @property
    def caplog(self) -> str:
        """Return captured log lines, if log capturing is enabled.

        .. versionadded:: 3.5
        """
        ...
    
    @property
    def capstdout(self) -> str:
        """Return captured text from stdout, if capturing is enabled.

        .. versionadded:: 3.0
        """
        ...
    
    @property
    def capstderr(self) -> str:
        """Return captured text from stderr, if capturing is enabled.

        .. versionadded:: 3.0
        """
        ...
    
    passed = ...
    failed = ...
    skipped = ...
    @property
    def fspath(self) -> str:
        ...
    
    @property
    def count_towards_summary(self) -> bool:
        """**Experimental** Whether this report should be counted towards the
        totals shown at the end of the test session: "1 passed, 1 failure, etc".

        .. note::

            This function is considered **experimental**, so beware that it is subject to changes
            even in patch releases.
        """
        ...
    
    @property
    def head_line(self) -> Optional[str]:
        """**Experimental** The head line shown with longrepr output for this
        report, more commonly during traceback representation during
        failures::

            ________ Test.foo ________


        In the example above, the head_line is "Test.foo".

        .. note::

            This function is considered **experimental**, so beware that it is subject to changes
            even in patch releases.
        """
        ...
    


@final
class TestReport(BaseReport):
    """Basic test report object (also used for setup and teardown calls if
    they fail)."""
    __test__ = ...
    def __init__(self, nodeid: str, location: Tuple[str, Optional[int], str], keywords, outcome: Literal[passed, failed, skipped], longrepr: Union[None, ExceptionInfo[BaseException], Tuple[str, int, str], str, TerminalRepr], when: Literal[setup, call, teardown], sections: Iterable[Tuple[str, str]] = ..., duration: float = ..., user_properties: Optional[Iterable[Tuple[str, object]]] = ..., **extra) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    @classmethod
    def from_item_and_call(cls, item: Item, call: CallInfo[None]) -> TestReport:
        """Create and fill a TestReport with standard item and call info."""
        ...
    


@final
class CollectReport(BaseReport):
    """Collection report object."""
    when = ...
    def __init__(self, nodeid: str, outcome: Literal[passed, skipped, failed], longrepr, result: Optional[List[Union[Item, Collector]]], sections: Iterable[Tuple[str, str]] = ..., **extra) -> None:
        ...
    
    @property
    def location(self): # -> tuple[str, None, str]:
        ...
    
    def __repr__(self) -> str:
        ...
    


class CollectErrorRepr(TerminalRepr):
    def __init__(self, msg: str) -> None:
        ...
    
    def toterminal(self, out: TerminalWriter) -> None:
        ...
    


def pytest_report_to_serializable(report: Union[CollectReport, TestReport]) -> Optional[Dict[str, Any]]:
    ...

def pytest_report_from_serializable(data: Dict[str, Any]) -> Optional[Union[CollectReport, TestReport]]:
    ...

