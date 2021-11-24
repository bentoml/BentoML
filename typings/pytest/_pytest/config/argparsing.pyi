
import argparse
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import py
from _pytest.compat import final
from typing_extensions import Literal

if TYPE_CHECKING:
    ...
FILE_OR_DIR = ...
@final
class Parser:
    """Parser for command line arguments and ini-file values.

    :ivar extra_info: Dict of generic param -> value to display in case
        there's an error processing the command line arguments.
    """
    prog: Optional[str] = ...
    def __init__(self, usage: Optional[str] = ..., processopt: Optional[Callable[[Argument], None]] = ...) -> None:
        ...
    
    def processoption(self, option: Argument) -> None:
        ...
    
    def getgroup(self, name: str, description: str = ..., after: Optional[str] = ...) -> OptionGroup:
        """Get (or create) a named option Group.

        :name: Name of the option group.
        :description: Long description for --help output.
        :after: Name of another group, used for ordering --help output.

        The returned group object has an ``addoption`` method with the same
        signature as :py:func:`parser.addoption
        <_pytest.config.argparsing.Parser.addoption>` but will be shown in the
        respective group in the output of ``pytest. --help``.
        """
        ...
    
    def addoption(self, *opts: str, **attrs: Any) -> None:
        """Register a command line option.

        :opts: Option names, can be short or long options.
        :attrs: Same attributes which the ``add_argument()`` function of the
           `argparse library <https://docs.python.org/library/argparse.html>`_
           accepts.

        After command line parsing, options are available on the pytest config
        object via ``config.option.NAME`` where ``NAME`` is usually set
        by passing a ``dest`` attribute, for example
        ``addoption("--long", dest="NAME", ...)``.
        """
        ...
    
    def parse(self, args: Sequence[Union[str, py.path.local]], namespace: Optional[argparse.Namespace] = ...) -> argparse.Namespace:
        ...
    
    def parse_setoption(self, args: Sequence[Union[str, py.path.local]], option: argparse.Namespace, namespace: Optional[argparse.Namespace] = ...) -> List[str]:
        ...
    
    def parse_known_args(self, args: Sequence[Union[str, py.path.local]], namespace: Optional[argparse.Namespace] = ...) -> argparse.Namespace:
        """Parse and return a namespace object with known arguments at this point."""
        ...
    
    def parse_known_and_unknown_args(self, args: Sequence[Union[str, py.path.local]], namespace: Optional[argparse.Namespace] = ...) -> Tuple[argparse.Namespace, List[str]]:
        """Parse and return a namespace object with known arguments, and
        the remaining arguments unknown at this point."""
        ...
    
    def addini(self, name: str, help: str, type: Optional[Literal[string, pathlist, args, linelist, bool]] = ..., default=...) -> None:
        """Register an ini-file option.

        :name: Name of the ini-variable.
        :type: Type of the variable, can be ``string``, ``pathlist``, ``args``,
               ``linelist`` or ``bool``.  Defaults to ``string`` if ``None`` or
               not passed.
        :default: Default value if no ini-file option exists but is queried.

        The value of ini-variables can be retrieved via a call to
        :py:func:`config.getini(name) <_pytest.config.Config.getini>`.
        """
        ...
    


class ArgumentError(Exception):
    """Raised if an Argument instance is created with invalid or
    inconsistent arguments."""
    def __init__(self, msg: str, option: Union[Argument, str]) -> None:
        ...
    
    def __str__(self) -> str:
        ...
    


class Argument:
    """Class that mimics the necessary behaviour of optparse.Option.

    It's currently a least effort implementation and ignoring choices
    and integer prefixes.

    https://docs.python.org/3/library/optparse.html#optparse-standard-option-types
    """
    _typ_map = ...
    def __init__(self, *names: str, **attrs: Any) -> None:
        """Store parms in private vars for use in add_argument."""
        ...
    
    def names(self) -> List[str]:
        ...
    
    def attrs(self) -> Mapping[str, Any]:
        ...
    
    def __repr__(self) -> str:
        ...
    


class OptionGroup:
    def __init__(self, name: str, description: str = ..., parser: Optional[Parser] = ...) -> None:
        ...
    
    def addoption(self, *optnames: str, **attrs: Any) -> None:
        """Add an option to this group.

        If a shortened version of a long option is specified, it will
        be suppressed in the help. addoption('--twowords', '--two-words')
        results in help showing '--two-words' only, but --twowords gets
        accepted **and** the automatic destination is in args.twowords.
        """
        ...
    


class MyOptionParser(argparse.ArgumentParser):
    def __init__(self, parser: Parser, extra_info: Optional[Dict[str, Any]] = ..., prog: Optional[str] = ...) -> None:
        ...
    
    def error(self, message: str) -> NoReturn:
        """Transform argparse error message into UsageError."""
        ...
    
    def parse_args(self, args: Optional[Sequence[str]] = ..., namespace: Optional[argparse.Namespace] = ...) -> argparse.Namespace:
        """Allow splitting of positional arguments."""
        ...
    
    if sys.version_info[: 2] < (3, 9):
        ...


class DropShorterLongHelpFormatter(argparse.HelpFormatter):
    """Shorten help for long options that differ only in extra hyphens.

    - Collapse **long** options that are the same except for extra hyphens.
    - Shortcut if there are only two options and one of them is a short one.
    - Cache result on the action object as this is called at least 2 times.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...
    


