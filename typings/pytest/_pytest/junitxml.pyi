
import xml.etree.ElementTree as ET
from typing import Callable, List, Optional, Union

import pytest
from _pytest._code.code import ExceptionRepr
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.reports import TestReport
from _pytest.terminal import TerminalReporter

"""Report test results in JUnit-XML format, for use with Jenkins and build
integration servers.

Based on initial code from Ross Lawley.

Output conforms to
https://github.com/jenkinsci/xunit-plugin/blob/master/src/main/resources/org/jenkinsci/plugins/xunit/types/model/xsd/junit-10.xsd
"""
xml_key = ...
def bin_xml_escape(arg: object) -> str:
    r"""Visually escape invalid XML characters.

    For example, transforms
        'hello\aworld\b'
    into
        'hello#x07world#x08'
    Note that the #xABs are *not* XML escapes - missing the ampersand &#xAB.
    The idea is to escape visually for the user rather than for XML itself.
    """
    ...

def merge_family(left, right) -> None:
    ...

families = ...
class _NodeReporter:
    def __init__(self, nodeid: Union[str, TestReport], xml: LogXML) -> None:
        ...
    
    def append(self, node: ET.Element) -> None:
        ...
    
    def add_property(self, name: str, value: object) -> None:
        ...
    
    def add_attribute(self, name: str, value: object) -> None:
        ...
    
    def make_properties_node(self) -> Optional[ET.Element]:
        """Return a Junit node containing custom properties, if any."""
        ...
    
    def record_testreport(self, testreport: TestReport) -> None:
        ...
    
    def to_xml(self) -> ET.Element:
        ...
    
    def write_captured_output(self, report: TestReport) -> None:
        ...
    
    def append_pass(self, report: TestReport) -> None:
        ...
    
    def append_failure(self, report: TestReport) -> None:
        ...
    
    def append_collect_error(self, report: TestReport) -> None:
        ...
    
    def append_collect_skipped(self, report: TestReport) -> None:
        ...
    
    def append_error(self, report: TestReport) -> None:
        ...
    
    def append_skipped(self, report: TestReport) -> None:
        ...
    
    def finalize(self) -> None:
        ...
    


@pytest.fixture
def record_property(request: FixtureRequest) -> Callable[[str, object], None]:
    """Add extra properties to the calling test.

    User properties become part of the test report and are available to the
    configured reporters, like JUnit XML.

    The fixture is callable with ``name, value``. The value is automatically
    XML-encoded.

    Example::

        def test_function(record_property):
            record_property("example_key", 1)
    """
    ...

@pytest.fixture
def record_xml_attribute(request: FixtureRequest) -> Callable[[str, object], None]:
    """Add extra xml attributes to the tag for the calling test.

    The fixture is callable with ``name, value``. The value is
    automatically XML-encoded.
    """
    ...

@pytest.fixture(scope="session")
def record_testsuite_property(request: FixtureRequest) -> Callable[[str, object], None]:
    """Record a new ``<property>`` tag as child of the root ``<testsuite>``.

    This is suitable to writing global information regarding the entire test
    suite, and is compatible with ``xunit2`` JUnit family.

    This is a ``session``-scoped fixture which is called with ``(name, value)``. Example:

    .. code-block:: python

        def test_foo(record_testsuite_property):
            record_testsuite_property("ARCH", "PPC")
            record_testsuite_property("STORAGE_TYPE", "CEPH")

    ``name`` must be a string, ``value`` will be converted to a string and properly xml-escaped.

    .. warning::

        Currently this fixture **does not work** with the
        `pytest-xdist <https://github.com/pytest-dev/pytest-xdist>`__ plugin. See issue
        `#7767 <https://github.com/pytest-dev/pytest/issues/7767>`__ for details.
    """
    ...

def pytest_addoption(parser: Parser) -> None:
    ...

def pytest_configure(config: Config) -> None:
    ...

def pytest_unconfigure(config: Config) -> None:
    ...

def mangle_test_address(address: str) -> List[str]:
    ...

class LogXML:
    def __init__(self, logfile, prefix: Optional[str], suite_name: str = ..., logging: str = ..., report_duration: str = ..., family=..., log_passing_tests: bool = ...) -> None:
        ...
    
    def finalize(self, report: TestReport) -> None:
        ...
    
    def node_reporter(self, report: Union[TestReport, str]) -> _NodeReporter:
        ...
    
    def add_stats(self, key: str) -> None:
        ...
    
    def pytest_runtest_logreport(self, report: TestReport) -> None:
        """Handle a setup/call/teardown report, generating the appropriate
        XML tags as necessary.

        Note: due to plugins like xdist, this hook may be called in interlaced
        order with reports from other nodes. For example:

        Usual call order:
            -> setup node1
            -> call node1
            -> teardown node1
            -> setup node2
            -> call node2
            -> teardown node2

        Possible call order in xdist:
            -> setup node1
            -> call node1
            -> setup node2
            -> call node2
            -> teardown node2
            -> teardown node1
        """
        ...
    
    def update_testcase_duration(self, report: TestReport) -> None:
        """Accumulate total duration for nodeid from given report and update
        the Junit.testcase with the new total if already created."""
        ...
    
    def pytest_collectreport(self, report: TestReport) -> None:
        ...
    
    def pytest_internalerror(self, excrepr: ExceptionRepr) -> None:
        ...
    
    def pytest_sessionstart(self) -> None:
        ...
    
    def pytest_sessionfinish(self) -> None:
        ...
    
    def pytest_terminal_summary(self, terminalreporter: TerminalReporter) -> None:
        ...
    
    def add_global_property(self, name: str, value: object) -> None:
        ...
    


