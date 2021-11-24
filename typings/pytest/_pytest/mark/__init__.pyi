
import warnings
from typing import TYPE_CHECKING, AbstractSet, Collection, List, Optional, Union

import attr
from _pytest.config import Config, ExitCode, UsageError, hookimpl
from _pytest.config.argparsing import Parser
from _pytest.deprecated import MINUS_K_COLON, MINUS_K_DASH
from _pytest.nodes import Item
from _pytest.store import StoreKey

from .expression import Expression, ParseError
from .structures import (
    EMPTY_PARAMETERSET_OPTION,
    MARK_GEN,
    Mark,
    MarkDecorator,
    MarkGenerator,
    ParameterSet,
    get_empty_parameterset_mark,
)

"""Generic mechanism for marking and selecting python functions."""
if TYPE_CHECKING:
    ...
__all__ = ["MARK_GEN", "Mark", "MarkDecorator", "MarkGenerator", "ParameterSet", "get_empty_parameterset_mark"]
old_mark_config_key = ...
def param(*values: object, marks: Union[MarkDecorator, Collection[Union[MarkDecorator, Mark]]] = ..., id: Optional[str] = ...) -> ParameterSet:
    """Specify a parameter in `pytest.mark.parametrize`_ calls or
    :ref:`parametrized fixtures <fixture-parametrize-marks>`.

    .. code-block:: python

        @pytest.mark.parametrize(
            "test_input,expected",
            [("3+5", 8), pytest.param("6*9", 42, marks=pytest.mark.xfail),],
        )
        def test_eval(test_input, expected):
            assert eval(test_input) == expected

    :param values: Variable args of the values of the parameter set, in order.
    :keyword marks: A single mark or a list of marks to be applied to this parameter set.
    :keyword str id: The id to attribute to this parameter set.
    """
    ...

def pytest_addoption(parser: Parser) -> None:
    ...

@hookimpl(tryfirst=True)
def pytest_cmdline_main(config: Config) -> Optional[Union[int, ExitCode]]:
    ...

@attr.s(slots=True)
class KeywordMatcher:
    """A matcher for keywords.

    Given a list of names, matches any substring of one of these names. The
    string inclusion check is case-insensitive.

    Will match on the name of colitem, including the names of its parents.
    Only matches names of items which are either a :class:`Class` or a
    :class:`Function`.

    Additionally, matches on names in the 'extra_keyword_matches' set of
    any item, as well as names directly assigned to test functions.
    """
    _names = ...
    @classmethod
    def from_item(cls, item: Item) -> KeywordMatcher:
        ...
    
    def __call__(self, subname: str) -> bool:
        ...
    


def deselect_by_keyword(items: List[Item], config: Config) -> None:
    ...

@attr.s(slots=True)
class MarkMatcher:
    """A matcher for markers which are present.

    Tries to match on any marker names, attached to the given colitem.
    """
    own_mark_names = ...
    @classmethod
    def from_item(cls, item) -> MarkMatcher:
        ...
    
    def __call__(self, name: str) -> bool:
        ...
    


def deselect_by_mark(items: List[Item], config: Config) -> None:
    ...

def pytest_collection_modifyitems(items: List[Item], config: Config) -> None:
    ...

def pytest_configure(config: Config) -> None:
    ...

def pytest_unconfigure(config: Config) -> None:
    ...

