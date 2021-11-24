
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    overload,
)

import py
from _pytest._code.code import ExceptionInfo, TerminalRepr, _TracebackStyle
from _pytest.compat import cached_property
from _pytest.config import Config
from _pytest.main import Session
from _pytest.mark.structures import Mark, MarkDecorator

if TYPE_CHECKING:
    ...
SEP = ...
tracebackcutdir = ...
def iterparentnodeids(nodeid: str) -> Iterator[str]:
    """Return the parent node IDs of a given node ID, inclusive.

    For the node ID

        "testing/code/test_excinfo.py::TestFormattedExcinfo::test_repr_source"

    the result would be

        ""
        "testing"
        "testing/code"
        "testing/code/test_excinfo.py"
        "testing/code/test_excinfo.py::TestFormattedExcinfo"
        "testing/code/test_excinfo.py::TestFormattedExcinfo::test_repr_source"

    Note that :: parts are only considered at the last / component.
    """
    ...

_NodeType = ...
class NodeMeta(type):
    def __call__(self, *k, **kw): # -> None:
        ...
    


class Node(metaclass=NodeMeta):
    """Base class for Collector and Item, the components of the test
    collection tree.

    Collector subclasses have children; Items are leaf nodes.
    """
    __slots__ = ...
    def __init__(self, name: str, parent: Optional[Node] = ..., config: Optional[Config] = ..., session: Optional[Session] = ..., fspath: Optional[py.path.local] = ..., nodeid: Optional[str] = ...) -> None:
        ...
    
    @classmethod
    def from_parent(cls, parent: Node, **kw): # -> Any:
        """Public constructor for Nodes.

        This indirection got introduced in order to enable removing
        the fragile logic from the node constructors.

        Subclasses can use ``super().from_parent(...)`` when overriding the
        construction.

        :param parent: The parent node of this Node.
        """
        ...
    
    @property
    def ihook(self): # -> FSHookProxy:
        """fspath-sensitive hook proxy used to call pytest hooks."""
        ...
    
    def __repr__(self) -> str:
        ...
    
    def warn(self, warning: Warning) -> None:
        """Issue a warning for this Node.

        Warnings will be displayed after the test session, unless explicitly suppressed.

        :param Warning warning:
            The warning instance to issue.

        :raises ValueError: If ``warning`` instance is not a subclass of Warning.

        Example usage:

        .. code-block:: python

            node.warn(PytestWarning("some message"))
            node.warn(UserWarning("some message"))

        .. versionchanged:: 6.2
            Any subclass of :class:`Warning` is now accepted, rather than only
            :class:`PytestWarning <pytest.PytestWarning>` subclasses.
        """
        ...
    
    @property
    def nodeid(self) -> str:
        """A ::-separated string denoting its collection tree address."""
        ...
    
    def __hash__(self) -> int:
        ...
    
    def setup(self) -> None:
        ...
    
    def teardown(self) -> None:
        ...
    
    def listchain(self) -> List[Node]:
        """Return list of all parent collectors up to self, starting from
        the root of collection tree."""
        ...
    
    def add_marker(self, marker: Union[str, MarkDecorator], append: bool = ...) -> None:
        """Dynamically add a marker object to the node.

        :param append:
            Whether to append the marker, or prepend it.
        """
        ...
    
    def iter_markers(self, name: Optional[str] = ...) -> Iterator[Mark]:
        """Iterate over all markers of the node.

        :param name: If given, filter the results by the name attribute.
        """
        ...
    
    def iter_markers_with_node(self, name: Optional[str] = ...) -> Iterator[Tuple[Node, Mark]]:
        """Iterate over all markers of the node.

        :param name: If given, filter the results by the name attribute.
        :returns: An iterator of (node, mark) tuples.
        """
        ...
    
    @overload
    def get_closest_marker(self, name: str) -> Optional[Mark]:
        ...
    
    @overload
    def get_closest_marker(self, name: str, default: Mark) -> Mark:
        ...
    
    def get_closest_marker(self, name: str, default: Optional[Mark] = ...) -> Optional[Mark]:
        """Return the first marker matching the name, from closest (for
        example function) to farther level (for example module level).

        :param default: Fallback return value if no marker was found.
        :param name: Name to filter by.
        """
        ...
    
    def listextrakeywords(self) -> Set[str]:
        """Return a set of all extra keywords in self and any parents."""
        ...
    
    def listnames(self) -> List[str]:
        ...
    
    def addfinalizer(self, fin: Callable[[], object]) -> None:
        """Register a function to be called when this node is finalized.

        This method can only be called when this node is active
        in a setup chain, for example during self.setup().
        """
        ...
    
    def getparent(self, cls: Type[_NodeType]) -> Optional[_NodeType]:
        """Get the next parent node (including self) which is an instance of
        the given class."""
        ...
    
    def repr_failure(self, excinfo: ExceptionInfo[BaseException], style: Optional[_TracebackStyle] = ...) -> Union[str, TerminalRepr]:
        """Return a representation of a collection or test failure.

        :param excinfo: Exception information for the failure.
        """
        ...
    


def get_fslocation_from_item(node: Node) -> Tuple[Union[str, py.path.local], Optional[int]]:
    """Try to extract the actual location from a node, depending on available attributes:

    * "location": a pair (path, lineno)
    * "obj": a Python object that the node wraps.
    * "fspath": just a path

    :rtype: A tuple of (str|py.path.local, int) with filename and line number.
    """
    ...

class Collector(Node):
    """Collector instances create children through collect() and thus
    iteratively build a tree."""
    class CollectError(Exception):
        """An error during collection, contains a custom message."""
        ...
    
    
    def collect(self) -> Iterable[Union[Item, Collector]]:
        """Return a list of children (items and collectors) for this
        collection node."""
        ...
    
    def repr_failure(self, excinfo: ExceptionInfo[BaseException]) -> Union[str, TerminalRepr]:
        """Return a representation of a collection failure.

        :param excinfo: Exception information for the failure.
        """
        ...
    


class FSCollector(Collector):
    def __init__(self, fspath: py.path.local, parent=..., config: Optional[Config] = ..., session: Optional[Session] = ..., nodeid: Optional[str] = ...) -> None:
        ...
    
    @classmethod
    def from_parent(cls, parent, *, fspath, **kw): # -> Any:
        """The public constructor."""
        ...
    
    def gethookproxy(self, fspath: py.path.local): # -> FSHookProxy:
        ...
    
    def isinitpath(self, path: py.path.local) -> bool:
        ...
    


class File(FSCollector):
    """Base class for collecting tests from a file.

    :ref:`non-python tests`.
    """
    ...


class Item(Node):
    """A basic test invocation item.

    Note that for a single function there might be multiple test invocation items.
    """
    nextitem = ...
    def __init__(self, name, parent=..., config: Optional[Config] = ..., session: Optional[Session] = ..., nodeid: Optional[str] = ...) -> None:
        ...
    
    def runtest(self) -> None:
        ...
    
    def add_report_section(self, when: str, key: str, content: str) -> None:
        """Add a new report section, similar to what's done internally to add
        stdout and stderr captured output::

            item.add_report_section("call", "stdout", "report section contents")

        :param str when:
            One of the possible capture states, ``"setup"``, ``"call"``, ``"teardown"``.
        :param str key:
            Name of the section, can be customized at will. Pytest uses ``"stdout"`` and
            ``"stderr"`` internally.
        :param str content:
            The full contents as a string.
        """
        ...
    
    def reportinfo(self) -> Tuple[Union[py.path.local, str], Optional[int], str]:
        ...
    
    @cached_property
    def location(self) -> Tuple[str, Optional[int], str]:
        ...
    


