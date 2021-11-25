

import typing
from typing import List, Text, Tuple

"""Useful functions for working with PyFilesystem paths.

This is broadly similar to the standard `os.path` module but works
with paths in the canonical format expected by all FS objects (that is,
separated by forward slashes and with an optional leading slash).

See :ref:`paths` for an explanation of PyFilesystem paths.

"""
if typing.TYPE_CHECKING: ...
__all__ = [
    "abspath",
    "basename",
    "combine",
    "dirname",
    "forcedir",
    "frombase",
    "isabs",
    "isbase",
    "isdotfile",
    "isparent",
    "issamedir",
    "iswildcard",
    "iteratepath",
    "join",
    "normpath",
    "parts",
    "recursepath",
    "relativefrom",
    "relpath",
    "split",
    "splitext",
]
_requires_normalization = ...

def normpath(path: Text) -> Text:
    """Normalize a path.

    This function simplifies a path by collapsing back-references
    and removing duplicated separators.

    Arguments:
        path (str): Path to normalize.

    Returns:
        str: A valid FS path.

    Example:
        >>> normpath("/foo//bar/frob/../baz")
        '/foo/bar/baz'
        >>> normpath("foo/../../bar")
        Traceback (most recent call last):
            ...
        fs.errors.IllegalBackReference: path 'foo/../../bar' contains back-references outside of filesystem

    """
    ...

def iteratepath(path: Text) -> List[Text]:
    """Iterate over the individual components of a path.

    Arguments:
        path (str): Path to iterate over.

    Returns:
        list: A list of path components.

    Example:
        >>> iteratepath('/foo/bar/baz')
        ['foo', 'bar', 'baz']

    """
    ...

def recursepath(path: Text, reverse: bool = ...) -> List[Text]:
    """Get intermediate paths from the root to the given path.

    Arguments:
        path (str): A PyFilesystem path
        reverse (bool): Reverses the order of the paths
            (default `False`).

    Returns:
        list: A list of paths.

    Example:
        >>> recursepath('a/b/c')
        ['/', '/a', '/a/b', '/a/b/c']

    """
    ...

def isabs(path: Text) -> bool:
    """Check if a path is an absolute path.

    Arguments:
        path (str): A PyFilesytem path.

    Returns:
        bool: `True` if the path is absolute (starts with a ``'/'``).

    """
    ...

def abspath(path: Text) -> Text:
    """Convert the given path to an absolute path.

    Since FS objects have no concept of a *current directory*, this
    simply adds a leading ``/`` character if the path doesn't already
    have one.

    Arguments:
        path (str): A PyFilesytem path.

    Returns:
        str: An absolute path.

    """
    ...

def relpath(path: Text) -> Text:
    """Convert the given path to a relative path.

    This is the inverse of `abspath`, stripping a leading ``'/'`` from
    the path if it is present.

    Arguments:
        path (str): A path to adjust.

    Returns:
        str: A relative path.

    Example:
        >>> relpath('/a/b')
        'a/b'

    """
    ...

def join(*paths: Text) -> Text:
    """Join any number of paths together.

    Arguments:
        *paths (str): Paths to join, given as positional arguments.

    Returns:
        str: The joined path.

    Example:
        >>> join('foo', 'bar', 'baz')
        'foo/bar/baz'
        >>> join('foo/bar', '../baz')
        'foo/baz'
        >>> join('foo/bar', '/baz')
        '/baz'

    """
    ...

def combine(path1: Text, path2: Text) -> Text:
    """Join two paths together.

    This is faster than :func:`~fs.path.join`, but only works when the
    second path is relative, and there are no back references in either
    path.

    Arguments:
        path1 (str): A PyFilesytem path.
        path2 (str): A PyFilesytem path.

    Returns:
        str: The joint path.

    Example:
        >>> combine("foo/bar", "baz")
        'foo/bar/baz'

    """
    ...

def parts(path: Text) -> List[Text]:
    """Split a path in to its component parts.

    Arguments:
        path (str): Path to split in to parts.

    Returns:
        list: List of components

    Example:
        >>> parts('/foo/bar/baz')
        ['/', 'foo', 'bar', 'baz']

    """
    ...

def split(path: Text) -> Tuple[Text, Text]:
    """Split a path into (head, tail) pair.

    This function splits a path into a pair (head, tail) where 'tail' is
    the last pathname component and 'head' is all preceding components.

    Arguments:
        path (str): Path to split

    Returns:
        (str, str): a tuple containing the head and the tail of the path.

    Example:
        >>> split("foo/bar")
        ('foo', 'bar')
        >>> split("foo/bar/baz")
        ('foo/bar', 'baz')
        >>> split("/foo/bar/baz")
        ('/foo/bar', 'baz')

    """
    ...

def splitext(path: Text) -> Tuple[Text, Text]:
    """Split the extension from the path.

    Arguments:
        path (str): A path to split.

    Returns:
        (str, str): A tuple containing the path and the extension.

    Example:
        >>> splitext('baz.txt')
        ('baz', '.txt')
        >>> splitext('foo/bar/baz.txt')
        ('foo/bar/baz', '.txt')
        >>> splitext('foo/bar/.foo')
        ('foo/bar/.foo', '')

    """
    ...

def isdotfile(path: Text) -> bool:
    """Detect if a path references a dot file.

    Arguments:
        path (str): Path to check.

    Returns:
        bool: `True` if the resource name starts with a ``'.'``.

    Example:
        >>> isdotfile('.baz')
        True
        >>> isdotfile('foo/bar/.baz')
        True
        >>> isdotfile('foo/bar.baz')
        False

    """
    ...

def dirname(path: Text) -> Text:
    """Return the parent directory of a path.

    This is always equivalent to the 'head' component of the value
    returned by ``split(path)``.

    Arguments:
        path (str): A PyFilesytem path.

    Returns:
        str: the parent directory of the given path.

    Example:
        >>> dirname('foo/bar/baz')
        'foo/bar'
        >>> dirname('/foo/bar')
        '/foo'
        >>> dirname('/foo')
        '/'

    """
    ...

def basename(path: Text) -> Text:
    """Return the basename of the resource referenced by a path.

    This is always equivalent to the 'tail' component of the value
    returned by split(path).

    Arguments:
        path (str): A PyFilesytem path.

    Returns:
        str: the name of the resource at the given path.

    Example:
        >>> basename('foo/bar/baz')
        'baz'
        >>> basename('foo/bar')
        'bar'
        >>> basename('foo/bar/')
        ''

    """
    ...

def issamedir(path1: Text, path2: Text) -> bool:
    """Check if two paths reference a resource in the same directory.

    Arguments:
        path1 (str): A PyFilesytem path.
        path2 (str): A PyFilesytem path.

    Returns:
        bool: `True` if the two resources are in the same directory.

    Example:
        >>> issamedir("foo/bar/baz.txt", "foo/bar/spam.txt")
        True
        >>> issamedir("foo/bar/baz/txt", "spam/eggs/spam.txt")
        False

    """
    ...

def isbase(path1: Text, path2: Text) -> bool:
    """Check if ``path1`` is a base of ``path2``.

    Arguments:
        path1 (str): A PyFilesytem path.
        path2 (str): A PyFilesytem path.

    Returns:
        bool: `True` if ``path2`` starts with ``path1``

    Example:
        >>> isbase('foo/bar', 'foo/bar/baz/egg.txt')
        True

    """
    ...

def isparent(path1: Text, path2: Text) -> bool:
    """Check if ``path1`` is a parent directory of ``path2``.

    Arguments:
        path1 (str): A PyFilesytem path.
        path2 (str): A PyFilesytem path.

    Returns:
        bool: `True` if ``path1`` is a parent directory of ``path2``

    Example:
        >>> isparent("foo/bar", "foo/bar/spam.txt")
        True
        >>> isparent("foo/bar/", "foo/bar")
        True
        >>> isparent("foo/barry", "foo/baz/bar")
        False
        >>> isparent("foo/bar/baz/", "foo/baz/bar")
        False

    """
    ...

def forcedir(path: Text) -> Text:
    """Ensure the path ends with a trailing forward slash.

    Arguments:
        path (str): A PyFilesytem path.

    Returns:
        str: The path, ending with a slash.

    Example:
        >>> forcedir("foo/bar")
        'foo/bar/'
        >>> forcedir("foo/bar/")
        'foo/bar/'
        >>> forcedir("foo/spam.txt")
        'foo/spam.txt/'

    """
    ...

def frombase(path1: Text, path2: Text) -> Text:
    """Get the final path of ``path2`` that isn't in ``path1``.

    Arguments:
        path1 (str): A PyFilesytem path.
        path2 (str): A PyFilesytem path.

    Returns:
        str: the final part of ``path2``.

    Example:
        >>> frombase('foo/bar/', 'foo/bar/baz/egg')
        'baz/egg'

    """
    ...

def relativefrom(base: Text, path: Text) -> Text:
    """Return a path relative from a given base path.

    Insert backrefs as appropriate to reach the path from the base.

    Arguments:
        base (str): Path to a directory.
        path (str): Path to make relative.

    Returns:
        str: the path to ``base`` from ``path``.

    >>> relativefrom("foo/bar", "baz/index.html")
    '../../baz/index.html'

    """
    ...

_WILD_CHARS = ...

def iswildcard(path: Text) -> bool:
    """Check if a path ends with a wildcard.

    Arguments:
        path (str): A PyFilesystem path.

    Returns:
        bool: `True` if path ends with a wildcard.

    Example:
        >>> iswildcard('foo/bar/baz.*')
        True
        >>> iswildcard('foo/bar')
        False

    """
    ...
