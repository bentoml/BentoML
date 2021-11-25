

from typing import Any, Iterator, List, NamedTuple, Optional, Pattern, Text, Tuple

from .base import FS
from .info import Info
from .lrucache import LRUCache

"""Useful functions for working with glob patterns.
"""

class GlobMatch(NamedTuple):
    path: Text
    info: Info

class Counts(NamedTuple):
    files: int
    directories: int
    data: int

class LineCounts(NamedTuple):
    lines: int
    non_blank: int

_PATTERN_CACHE: LRUCache[Tuple[Text, bool], Tuple[int, bool, Pattern[Any]]] = ...

def match(pattern: str, path: str) -> bool:
    """Compare a glob pattern with a path (case sensitive).

    Arguments:
        pattern (str): A glob pattern.
        path (str): A path.

    Returns:
        bool: ``True`` if the path matches the pattern.

    Example:

        >>> from fs.glob import match
        >>> match("**/*.py", "/fs/glob.py")
        True

    """
    ...

def imatch(pattern: str, path: str) -> bool:
    """Compare a glob pattern with a path (case insensitive).

    Arguments:
        pattern (str): A glob pattern.
        path (str): A path.

    Returns:
        bool: ``True`` if the path matches the pattern.

    """
    ...

class Globber:
    """A generator of glob results."""

    def __init__(
        self,
        fs: FS,
        pattern: str,
        path: str = ...,
        namespaces: Optional[List[str]] = ...,
        case_sensitive: bool = ...,
        exclude_dirs: Optional[List[str]] = ...,
    ) -> None:
        """Create a new Globber instance.

        Arguments:
            fs (~fs.base.FS): A filesystem object
            pattern (str): A glob pattern, e.g. ``"**/*.py"``
            path (str): A path to a directory in the filesystem.
            namespaces (list): A list of additional info namespaces.
            case_sensitive (bool): If ``True``, the path matching will be
                case *sensitive* i.e. ``"FOO.py"`` and ``"foo.py"`` will be
                different, otherwise path matching will be case *insensitive*.
            exclude_dirs (list): A list of patterns to exclude when searching,
                e.g. ``["*.git"]``.

        """
        ...
    def __repr__(self) -> Text: ...
    def __iter__(self) -> Iterator[GlobMatch]:
        """Get an iterator of :class:`fs.glob.GlobMatch` objects."""
        ...
    def count(self) -> Counts:
        """Count files / directories / data in matched paths.

        Example:
            >>> my_fs.glob('**/*.py').count()
            Counts(files=2, directories=0, data=55)

        Returns:
            `~Counts`: A named tuple containing results.

        """
        ...
    def count_lines(self) -> LineCounts:
        """Count the lines in the matched files.

        Returns:
            `~LineCounts`: A named tuple containing line counts.

        Example:
            >>> my_fs.glob('**/*.py').count_lines()
            LineCounts(lines=4, non_blank=3)

        """
        ...
    def remove(self) -> int:
        """Remove all matched paths.

        Returns:
            int: Number of file and directories removed.

        Example:
            >>> my_fs.glob('**/*.pyc').remove()
            2

        """
        ...

class BoundGlobber:
    """A `~fs.glob.Globber` object bound to a filesystem.

    An instance of this object is available on every Filesystem object
    as the `~fs.base.FS.glob` property.

    """

    __slots__ = ["fs"]
    def __init__(self, fs: FS) -> None:
        """Create a new bound Globber.

        Arguments:
            fs (FS): A filesystem object to bind to.

        """
        ...
    def __repr__(self) -> Text: ...
    def __call__(
        self,
        pattern: str,
        path: str = ...,
        namespaces: Optional[List[str]] = ...,
        case_sensitive: bool = ...,
        exclude_dirs: Optional[List[str]] = ...,
    ) -> Globber:
        """Match resources on the bound filesystem againsts a glob pattern.

        Arguments:
            pattern (str): A glob pattern, e.g. ``"**/*.py"``
            namespaces (list): A list of additional info namespaces.
            case_sensitive (bool): If ``True``, the path matching will be
                case *sensitive* i.e. ``"FOO.py"`` and ``"foo.py"`` will
                be different, otherwise path matching will be case **insensitive**.
            exclude_dirs (list): A list of patterns to exclude when searching,
                e.g. ``["*.git"]``.

        Returns:
            `Globber`: An object that may be queried for the glob matches.

        """
        ...
