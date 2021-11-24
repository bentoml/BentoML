
from typing import Generator, Optional, Tuple

import attr
from _pytest.config import Config, hookimpl
from _pytest.config.argparsing import Parser
from _pytest.mark.structures import Mark
from _pytest.nodes import Item
from _pytest.reports import BaseReport
from _pytest.runner import CallInfo

"""Support for skip/xfail functions and markers."""
def pytest_addoption(parser: Parser) -> None:
    ...

def pytest_configure(config: Config) -> None:
    ...

def evaluate_condition(item: Item, mark: Mark, condition: object) -> Tuple[bool, str]:
    """Evaluate a single skipif/xfail condition.

    If an old-style string condition is given, it is eval()'d, otherwise the
    condition is bool()'d. If this fails, an appropriately formatted pytest.fail
    is raised.

    Returns (result, reason). The reason is only relevant if the result is True.
    """
    ...

@attr.s(slots=True, frozen=True)
class Skip:
    """The result of evaluate_skip_marks()."""
    reason = ...


def evaluate_skip_marks(item: Item) -> Optional[Skip]:
    """Evaluate skip and skipif marks on item, returning Skip if triggered."""
    ...

@attr.s(slots=True, frozen=True)
class Xfail:
    """The result of evaluate_xfail_marks()."""
    reason = ...
    run = ...
    strict = ...
    raises = ...


def evaluate_xfail_marks(item: Item) -> Optional[Xfail]:
    """Evaluate xfail marks on item, returning Xfail if triggered."""
    ...

skipped_by_mark_key = ...
xfailed_key = ...
unexpectedsuccess_key = ...
@hookimpl(tryfirst=True)
def pytest_runtest_setup(item: Item) -> None:
    ...

@hookimpl(hookwrapper=True)
def pytest_runtest_call(item: Item) -> Generator[None, None, None]:
    ...

@hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: Item, call: CallInfo[None]): # -> Generator[None, None, None]:
    ...

def pytest_report_teststatus(report: BaseReport) -> Optional[Tuple[str, str, str]]:
    ...

