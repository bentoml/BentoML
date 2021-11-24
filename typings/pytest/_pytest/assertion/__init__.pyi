
import sys
from typing import TYPE_CHECKING, Any, Generator, List, Optional

from _pytest.assertion import rewrite, truncate, util
from _pytest.assertion.rewrite import assertstate_key
from _pytest.config import Config, hookimpl
from _pytest.config.argparsing import Parser
from _pytest.main import Session
from _pytest.nodes import Item

"""Support for presenting detailed information in failing assertions."""
if TYPE_CHECKING:
    ...
def pytest_addoption(parser: Parser) -> None:
    ...

def register_assert_rewrite(*names: str) -> None:
    """Register one or more module names to be rewritten on import.

    This function will make sure that this module or all modules inside
    the package will get their assert statements rewritten.
    Thus you should make sure to call this before the module is
    actually imported, usually in your __init__.py if you are a plugin
    using a package.

    :raises TypeError: If the given module names are not strings.
    """
    ...

class DummyRewriteHook:
    """A no-op import hook for when rewriting is disabled."""
    def mark_rewrite(self, *names: str) -> None:
        ...
    


class AssertionState:
    """State for the assertion plugin."""
    def __init__(self, config: Config, mode) -> None:
        ...
    


def install_importhook(config: Config) -> rewrite.AssertionRewritingHook:
    """Try to install the rewrite hook, raise SystemError if it fails."""
    ...

def pytest_collection(session: Session) -> None:
    ...

@hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_protocol(item: Item) -> Generator[None, None, None]:
    """Setup the pytest_assertrepr_compare and pytest_assertion_pass hooks.

    The rewrite module will use util._reprcompare if it exists to use custom
    reporting via the pytest_assertrepr_compare hook.  This sets up this custom
    comparison for the test.
    """
    ...

def pytest_sessionfinish(session: Session) -> None:
    ...

def pytest_assertrepr_compare(config: Config, op: str, left: Any, right: Any) -> Optional[List[str]]:
    ...

