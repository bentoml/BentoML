

import typing
from typing import Callable, Iterable, Pattern, Text, Tuple

from .lrucache import LRUCache

"""Match wildcard filenames.
"""
if typing.TYPE_CHECKING: ...
_PATTERN_CACHE: LRUCache[Tuple[Text, bool], Pattern] = ...

def match(pattern: Text, name: Text) -> bool:
    """Test whether a name matches a wildcard pattern.

    Arguments:
        pattern (str): A wildcard pattern, e.g. ``"*.py"``.
        name (str): A filename.

    Returns:
        bool: `True` if the filename matches the pattern.

    """
    ...

def imatch(pattern: Text, name: Text) -> bool:
    """Test whether a name matches a wildcard pattern (case insensitive).

    Arguments:
        pattern (str): A wildcard pattern, e.g. ``"*.py"``.
        name (bool): A filename.

    Returns:
        bool: `True` if the filename matches the pattern.

    """
    ...

def match_any(patterns: Iterable[Text], name: Text) -> bool:
    """Test if a name matches any of a list of patterns.

    Will return `True` if ``patterns`` is an empty list.

    Arguments:
        patterns (list): A list of wildcard pattern, e.g ``["*.py",
            "*.pyc"]``
        name (str): A filename.

    Returns:
        bool: `True` if the name matches at least one of the patterns.

    """
    ...

def imatch_any(patterns: Iterable[Text], name: Text) -> bool:
    """Test if a name matches any of a list of patterns (case insensitive).

    Will return `True` if ``patterns`` is an empty list.

    Arguments:
        patterns (list): A list of wildcard pattern, e.g ``["*.py",
            "*.pyc"]``
        name (str): A filename.

    Returns:
        bool: `True` if the name matches at least one of the patterns.

    """
    ...

def get_matcher(
    patterns: Iterable[Text], case_sensitive: bool
) -> Callable[[Text], bool]:
    """Get a callable that matches names against the given patterns.

    Arguments:
        patterns (list): A list of wildcard pattern. e.g. ``["*.py",
            "*.pyc"]``
        case_sensitive (bool): If ``True``, then the callable will be case
            sensitive, otherwise it will be case insensitive.

    Returns:
        callable: a matcher that will return `True` if the name given as
        an argument matches any of the given patterns.

    Example:
        >>> from fs import wildcard
        >>> is_python = wildcard.get_matcher(['*.py'], True)
        >>> is_python('__init__.py')
        True
        >>> is_python('foo.txt')
        False

    """
    ...
