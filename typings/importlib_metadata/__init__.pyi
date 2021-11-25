

import abc
import collections
import csv
import email
import functools
import itertools
import operator
import os
import pathlib
import posixpath
import re
import sys
import textwrap
import warnings
from contextlib import suppress
from importlib import import_module
from importlib.abc import MetaPathFinder
from itertools import starmap
from typing import Any, List, Mapping, Optional, Union

import zipp

from . import _adapters, _meta
from ._collections import FreezableDefaultDict, Pair
from ._compat import NullFinder, PyPy_repr, install
from ._functools import method_cache
from ._itertools import unique_everseen
from ._meta import PackageMetadata, SimplePath

__all__ = ['Distribution', 'DistributionFinder', 'PackageMetadata', 'PackageNotFoundError', 'distribution', 'distributions', 'entry_points', 'files', 'metadata', 'packages_distributions', 'requires', 'version']
class PackageNotFoundError(ModuleNotFoundError):
    """The package was not found."""
    def __str__(self) -> str:
        ...
    
    @property
    def name(self): # -> Any:
        ...
    


class Sectioned:
    """
    A simple entry point config parser for performance

    >>> for item in Sectioned.read(Sectioned._sample):
    ...     print(item)
    Pair(name='sec1', value='# comments ignored')
    Pair(name='sec1', value='a = 1')
    Pair(name='sec1', value='b = 2')
    Pair(name='sec2', value='a = 2')

    >>> res = Sectioned.section_pairs(Sectioned._sample)
    >>> item = next(res)
    >>> item.name
    'sec1'
    >>> item.value
    Pair(name='a', value='1')
    >>> item = next(res)
    >>> item.value
    Pair(name='b', value='2')
    >>> item = next(res)
    >>> item.name
    'sec2'
    >>> item.value
    Pair(name='a', value='2')
    >>> list(res)
    []
    """
    _sample = ...
    @classmethod
    def section_pairs(cls, text): # -> Generator[Pair, None, None]:
        ...
    
    @staticmethod
    def read(text, filter_=...): # -> Generator[Pair, None, None]:
        ...
    
    @staticmethod
    def valid(line): # -> bool:
        ...
    


class EntryPoint(PyPy_repr, collections.namedtuple('EntryPointBase', 'name value group')):
    """An entry point as defined by Python packaging conventions.

    See `the packaging docs on entry points
    <https://packaging.python.org/specifications/entry-points/>`_
    for more information.
    """
    pattern = ...
    dist: Optional[Distribution] = ...
    def load(self): # -> ModuleType:
        """Load the entry point from its definition. If only a module
        is indicated by the value, return that module. Otherwise,
        return the named object.
        """
        ...
    
    @property
    def module(self): # -> str | Any:
        ...
    
    @property
    def attr(self): # -> str | Any:
        ...
    
    @property
    def extras(self): # -> list[Match[str]]:
        ...
    
    def __iter__(self): # -> Iterator[Unknown | Self@EntryPoint]:
        """
        Supply iter so one may construct dicts of EntryPoints by name.
        """
        ...
    
    def __reduce__(self): # -> tuple[Type[EntryPointBase], tuple[Unknown, Unknown, Unknown]]:
        ...
    
    def matches(self, **params): # -> bool:
        ...
    


class DeprecatedList(list):
    """
    Allow an otherwise immutable object to implement mutability
    for compatibility.

    >>> recwarn = getfixture('recwarn')
    >>> dl = DeprecatedList(range(3))
    >>> dl[0] = 1
    >>> dl.append(3)
    >>> del dl[3]
    >>> dl.reverse()
    >>> dl.sort()
    >>> dl.extend([4])
    >>> dl.pop(-1)
    4
    >>> dl.remove(1)
    >>> dl += [5]
    >>> dl + [6]
    [1, 2, 5, 6]
    >>> dl + (6,)
    [1, 2, 5, 6]
    >>> dl.insert(0, 0)
    >>> dl
    [0, 1, 2, 5]
    >>> dl == [0, 1, 2, 5]
    True
    >>> dl == (0, 1, 2, 5)
    True
    >>> len(recwarn)
    1
    """
    _warn = ...
    def __setitem__(self, *args, **kwargs): # -> None:
        ...
    
    def __delitem__(self, *args, **kwargs): # -> None:
        ...
    
    def append(self, *args, **kwargs): # -> None:
        ...
    
    def reverse(self, *args, **kwargs): # -> None:
        ...
    
    def extend(self, *args, **kwargs): # -> None:
        ...
    
    def pop(self, *args, **kwargs):
        ...
    
    def remove(self, *args, **kwargs): # -> None:
        ...
    
    def __iadd__(self, *args, **kwargs): # -> DeprecatedList:
        ...
    
    def __add__(self, other): # -> Self@DeprecatedList:
        ...
    
    def insert(self, *args, **kwargs): # -> None:
        ...
    
    def sort(self, *args, **kwargs): # -> None:
        ...
    
    def __eq__(self, other) -> bool:
        ...
    


class EntryPoints(DeprecatedList):
    """
    An immutable collection of selectable EntryPoint objects.
    """
    __slots__ = ...
    def __getitem__(self, name):
        """
        Get the EntryPoint in self matching name.
        """
        ...
    
    def select(self, **params): # -> EntryPoints:
        """
        Select entry points from self that match the
        given parameters (typically group and/or name).
        """
        ...
    
    @property
    def names(self): # -> set[Unknown]:
        """
        Return the set of all names of all entry points.
        """
        ...
    
    @property
    def groups(self): # -> set[Unknown]:
        """
        Return the set of all groups of all entry points.

        For coverage while SelectableGroups is present.
        >>> EntryPoints().groups
        set()
        """
        ...
    


class Deprecated:
    """
    Compatibility add-in for mapping to indicate that
    mapping behavior is deprecated.

    >>> recwarn = getfixture('recwarn')
    >>> class DeprecatedDict(Deprecated, dict): pass
    >>> dd = DeprecatedDict(foo='bar')
    >>> dd.get('baz', None)
    >>> dd['foo']
    'bar'
    >>> list(dd)
    ['foo']
    >>> list(dd.keys())
    ['foo']
    >>> 'foo' in dd
    True
    >>> list(dd.values())
    ['bar']
    >>> len(recwarn)
    1
    """
    _warn = ...
    def __getitem__(self, name):
        ...
    
    def get(self, name, default=...):
        ...
    
    def __iter__(self):
        ...
    
    def __contains__(self, *args):
        ...
    
    def keys(self):
        ...
    
    def values(self):
        ...
    


class SelectableGroups(Deprecated, dict):
    """
    A backward- and forward-compatible result from
    entry_points that fully implements the dict interface.
    """
    @classmethod
    def load(cls, eps): # -> Self@SelectableGroups:
        ...
    
    @property
    def groups(self): # -> set[Unknown]:
        ...
    
    @property
    def names(self): # -> set[Unknown]:
        """
        for coverage:
        >>> SelectableGroups().names
        set()
        """
        ...
    
    def select(self, **params): # -> Self@SelectableGroups | EntryPoints:
        ...
    


class PackagePath(pathlib.PurePosixPath):
    """A reference to a path in a package"""
    def read_text(self, encoding=...):
        ...
    
    def read_binary(self):
        ...
    
    def locate(self):
        """Return a path-like object for this path"""
        ...
    


class FileHash:
    def __init__(self, spec) -> None:
        ...
    
    def __repr__(self): # -> str:
        ...
    


class Distribution:
    """A Python distribution package."""
    @abc.abstractmethod
    def read_text(self, filename): # -> None:
        """Attempt to load metadata file given by the name.

        :param filename: The name of the file in the distribution info.
        :return: The text if found, otherwise None.
        """
        ...
    
    @abc.abstractmethod
    def locate_file(self, path): # -> None:
        """
        Given a path to a file in this distribution, return a path
        to it.
        """
        ...
    
    @classmethod
    def from_name(cls, name): # -> Any:
        """Return the Distribution for the given package name.

        :param name: The name of the distribution package to search for.
        :return: The Distribution instance (or subclass thereof) for the named
            package, if found.
        :raises PackageNotFoundError: When the named package's distribution
            metadata cannot be found.
        """
        ...
    
    @classmethod
    def discover(cls, **kwargs): # -> Iterator[Any]:
        """Return an iterable of Distribution objects for all packages.

        Pass a ``context`` or pass keyword arguments for constructing
        a context.

        :context: A ``DistributionFinder.Context`` object.
        :return: Iterable of Distribution objects for all packages.
        """
        ...
    
    @staticmethod
    def at(path): # -> PathDistribution:
        """Return a Distribution for the indicated metadata path

        :param path: a string or path-like object
        :return: a concrete Distribution instance for the path
        """
        ...
    
    @property
    def metadata(self) -> _meta.PackageMetadata:
        """Return the parsed metadata for this Distribution.

        The returned object will have keys that name the various bits of
        metadata.  See PEP 566 for details.
        """
        ...
    
    @property
    def name(self): # -> str:
        """Return the 'Name' metadata for the distribution package."""
        ...
    
    @property
    def version(self): # -> str:
        """Return the 'Version' metadata for the distribution package."""
        ...
    
    @property
    def entry_points(self): # -> EntryPoints:
        ...
    
    @property
    def files(self): # -> None:
        """Files in this distribution.

        :return: List of PackagePath for this distribution or None

        Result is `None` if the metadata file that enumerates files
        (i.e. RECORD for dist-info or SOURCES.txt for egg-info) is
        missing.
        Result may be empty if the metadata exists but is empty.
        """
        ...
    
    @property
    def requires(self): # -> List[Any] | None:
        """Generated requirements specified for this Distribution"""
        ...
    


class DistributionFinder(MetaPathFinder):
    """
    A MetaPathFinder capable of discovering installed distributions.
    """
    class Context:
        """
        Keyword arguments presented by the caller to
        ``distributions()`` or ``Distribution.discover()``
        to narrow the scope of a search for distributions
        in all DistributionFinders.

        Each DistributionFinder may expect any parameters
        and should attempt to honor the canonical
        parameters defined below when appropriate.
        """
        name = ...
        def __init__(self, **kwargs) -> None:
            ...
        
        @property
        def path(self): # -> Any | list[str]:
            """
            The sequence of directory path that a distribution finder
            should search.

            Typically refers to Python installed package paths such as
            "site-packages" directories and defaults to ``sys.path``.
            """
            ...
        
    
    
    @abc.abstractmethod
    def find_distributions(self, context=...): # -> None:
        """
        Find distributions.

        Return an iterable of all Distribution instances capable of
        loading the metadata for packages matching the ``context``,
        a DistributionFinder.Context instance.
        """
        ...
    


class FastPath:
    """
    Micro-optimized class for searching a path for
    children.
    """
    @functools.lru_cache()
    def __new__(cls, root): # -> Self@FastPath:
        ...
    
    def __init__(self, root) -> None:
        ...
    
    def joinpath(self, child): # -> Path:
        ...
    
    def children(self): # -> list[str] | dict[Unknown, Any | None]:
        ...
    
    def zip_children(self): # -> dict[Unknown, Any | None]:
        ...
    
    def search(self, name): # -> chain[Unknown]:
        ...
    
    @property
    def mtime(self): # -> float | None:
        ...
    
    @method_cache
    def lookup(self, mtime): # -> Lookup:
        ...
    


class Lookup:
    def __init__(self, path: FastPath) -> None:
        ...
    
    def search(self, prepared): # -> chain[Unknown]:
        ...
    


class Prepared:
    """
    A prepared search for metadata on a possibly-named package.
    """
    normalized = ...
    legacy_normalized = ...
    def __init__(self, name) -> None:
        ...
    
    @staticmethod
    def normalize(name): # -> str:
        """
        PEP 503 normalization plus dashes as underscores.
        """
        ...
    
    @staticmethod
    def legacy_normalize(name):
        """
        Normalize the package name as found in the convention in
        older packaging tools versions and specs.
        """
        ...
    
    def __bool__(self): # -> bool:
        ...
    


@install
class MetadataPathFinder(NullFinder, DistributionFinder):
    """A degenerate finder for distribution packages on the file system.

    This finder supplies only a find_distributions() method for versions
    of Python that do not have a PathFinder find_distributions().
    """
    def find_distributions(self, context=...): # -> map[PathDistribution]:
        """
        Find distributions.

        Return an iterable of all Distribution instances capable of
        loading the metadata for packages matching ``context.name``
        (or all names if ``None`` indicated) along the paths in the list
        of directories ``context.path``.
        """
        ...
    
    def invalidate_caches(cls): # -> None:
        ...
    


class PathDistribution(Distribution):
    def __init__(self, path: SimplePath) -> None:
        """Construct a distribution.

        :param path: SimplePath indicating the metadata directory.
        """
        ...
    
    def read_text(self, filename): # -> None:
        ...
    
    def locate_file(self, path):
        ...
    


def distribution(distribution_name): # -> Any:
    """Get the ``Distribution`` instance for the named package.

    :param distribution_name: The name of the distribution package as a string.
    :return: A ``Distribution`` instance (or subclass thereof).
    """
    ...

def distributions(**kwargs): # -> Iterator[Any]:
    """Get all ``Distribution`` instances in the current environment.

    :return: An iterable of ``Distribution`` instances.
    """
    ...

def metadata(distribution_name) -> _meta.PackageMetadata:
    """Get the metadata for the named package.

    :param distribution_name: The name of the distribution package to query.
    :return: A PackageMetadata containing the parsed metadata.
    """
    ...

def version(distribution_name: str) -> str:
    """Get the version string for the named package.

    :param distribution_name: The name of the distribution package to query.
    :return: The version string for the package as defined in the package's
        "Version" metadata key.
    """
    ...

def entry_points(**params) -> Union[EntryPoints, SelectableGroups]:
    """Return EntryPoint objects for all installed packages.

    Pass selection parameters (group or name) to filter the
    result to entry points matching those properties (see
    EntryPoints.select()).

    For compatibility, returns ``SelectableGroups`` object unless
    selection parameters are supplied. In the future, this function
    will return ``EntryPoints`` instead of ``SelectableGroups``
    even when no selection parameters are supplied.

    For maximum future compatibility, pass selection parameters
    or invoke ``.select`` with parameters on the result.

    :return: EntryPoints or SelectableGroups for all installed packages.
    """
    ...

def files(distribution_name): # -> Any:
    """Return a list of files for the named package.

    :param distribution_name: The name of the distribution package to query.
    :return: List of files composing the distribution.
    """
    ...

def requires(distribution_name): # -> Any:
    """
    Return a list of requirements for the named package.

    :return: An iterator of requirements, suitable for
        packaging.requirement.Requirement.
    """
    ...

def packages_distributions() -> Mapping[str, List[str]]:
    """
    Return a mapping of top-level packages to their
    distributions.

    >>> import collections.abc
    >>> pkgs = packages_distributions()
    >>> all(isinstance(dist, collections.abc.Sequence) for dist in pkgs.values())
    True
    """
    ...

