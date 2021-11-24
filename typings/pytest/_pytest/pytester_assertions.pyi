
from typing import Dict, Sequence, Tuple, Union

from _pytest.reports import CollectReport, TestReport

"""Helper plugin for pytester; should not be loaded on its own."""
def assertoutcome(outcomes: Tuple[Sequence[TestReport], Sequence[Union[CollectReport, TestReport]], Sequence[Union[CollectReport, TestReport]]],, passed: int = ..., skipped: int = ..., failed: int = ...) -> None:
    ...

def assert_outcomes(outcomes: Dict[str, int], passed: int = ..., skipped: int = ..., failed: int = ..., errors: int = ..., xpassed: int = ..., xfailed: int = ...) -> None:
    """Assert that the specified outcomes appear with the respective
    numbers (0 means it didn't occur) in the text output from a test run."""
    ...

