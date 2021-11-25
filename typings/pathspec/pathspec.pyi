

from os import PathLike
from typing import (
    Any,
    AnyStr,
    Callable,
    Collection,
    Iterable,
    Iterator,
    Optional,
    Text,
    Union,
)

from .pattern import Pattern
from .util import TreeEntry

"""
This module provides an object oriented interface for pattern matching
of files.
"""

class PathSpec:
    """
    The :class:`PathSpec` class is a wrapper around a list of compiled
    :class:`.Pattern` instances.
    """

    def __init__(self, patterns: Iterable[Pattern]) -> None:
        """
        Initializes the :class:`PathSpec` instance.

        *patterns* (:class:`~collections.abc.Collection` or :class:`~collections.abc.Iterable`)
        yields each compiled pattern (:class:`.Pattern`).
        """
        ...
    def __eq__(self, other: PathSpec) -> bool:
        """
        Tests the equality of this path-spec with *other* (:class:`PathSpec`)
        by comparing their :attr:`~PathSpec.patterns` attributes.
        """
        ...
    def __len__(self):  # -> int:
        """
        Returns the number of compiled patterns this path-spec contains
        (:class:`int`).
        """
        ...
    def __add__(self, other: PathSpec) -> PathSpec:
        """
        Combines the :attr:`Pathspec.patterns` patterns from two
        :class:`PathSpec` instances.
        """
        ...
    def __iadd__(self, other: PathSpec) -> PathSpec:
        """
        Adds the :attr:`Pathspec.patterns` patterns from one :class:`PathSpec`
        instance to this instance.
        """
        ...
    @classmethod
    def from_lines(
        cls,
        pattern_factory: Union[Text, Callable[[AnyStr], Pattern]],
        lines: Iterable[AnyStr],
    ) -> PathSpec:
        """
        Compiles the pattern lines.

        *pattern_factory* can be either the name of a registered pattern
        factory (:class:`str`), or a :class:`~collections.abc.Callable` used
        to compile patterns. It must accept an uncompiled pattern (:class:`str`)
        and return the compiled pattern (:class:`.Pattern`).

        *lines* (:class:`~collections.abc.Iterable`) yields each uncompiled
        pattern (:class:`str`). This simply has to yield each line so it can
        be a :class:`file` (e.g., from :func:`open` or :class:`io.StringIO`)
        or the result from :meth:`str.splitlines`.

        Returns the :class:`PathSpec` instance.
        """
        ...
    def match_file(
        self,
        file: Union[Text, PathLike[Any]],
        separators: Optional[Collection[Text]] = ...,
    ) -> bool:
        """
        Matches the file to this path-spec.

        *file* (:class:`str` or :class:`~pathlib.PurePath`) is the file path
        to be matched against :attr:`self.patterns <PathSpec.patterns>`.

        *separators* (:class:`~collections.abc.Collection` of :class:`str`)
        optionally contains the path separators to normalize. See
        :func:`~pathspec.util.normalize_file` for more information.

        Returns :data:`True` if *file* matched; otherwise, :data:`False`.
        """
        ...
    def match_entries(
        self, entries: Iterable[TreeEntry], separators: Optional[Collection[Text]] = ...
    ) -> Iterator[TreeEntry]:
        """
        Matches the entries to this path-spec.

        *entries* (:class:`~collections.abc.Iterable` of :class:`~util.TreeEntry`)
        contains the entries to be matched against :attr:`self.patterns <PathSpec.patterns>`.

        *separators* (:class:`~collections.abc.Collection` of :class:`str`;
        or :data:`None`) optionally contains the path separators to
        normalize. See :func:`~pathspec.util.normalize_file` for more
        information.

        Returns the matched entries (:class:`~collections.abc.Iterator` of
        :class:`~util.TreeEntry`).
        """
        ...
    def match_files(
        self,
        files: Iterable[Union[Text, PathLike]],
        separators: Optional[Collection[Text]] = ...,
    ) -> Iterator[Union[Text, PathLike]]:
        """
        Matches the files to this path-spec.

        *files* (:class:`~collections.abc.Iterable` of :class:`str; or
        :class:`pathlib.PurePath`) contains the file paths to be matched
        against :attr:`self.patterns <PathSpec.patterns>`.

        *separators* (:class:`~collections.abc.Collection` of :class:`str`;
        or :data:`None`) optionally contains the path separators to
        normalize. See :func:`~pathspec.util.normalize_file` for more
        information.

        Returns the matched files (:class:`~collections.abc.Iterator` of
        :class:`str` or :class:`pathlib.PurePath`).
        """
        ...
    def match_tree_entries(
        self,
        root: Text,
        on_error: Optional[Callable] = ...,
        follow_links: Optional[bool] = ...,
    ) -> Iterator[TreeEntry]:
        """
        Walks the specified root path for all files and matches them to this
        path-spec.

        *root* (:class:`str`; or :class:`pathlib.PurePath`) is the root
        directory to search.

        *on_error* (:class:`~collections.abc.Callable` or :data:`None`)
        optionally is the error handler for file-system exceptions. See
        :func:`~pathspec.util.iter_tree_entries` for more information.

        *follow_links* (:class:`bool` or :data:`None`) optionally is whether
        to walk symbolic links that resolve to directories. See
        :func:`~pathspec.util.iter_tree_files` for more information.

        Returns the matched files (:class:`~collections.abc.Iterator` of
        :class:`.TreeEntry`).
        """
        ...
    def match_tree_files(
        self,
        root: Text,
        on_error: Optional[Callable] = ...,
        follow_links: Optional[bool] = ...,
    ) -> Iterator[Text]:
        """
        Walks the specified root path for all files and matches them to this
        path-spec.

        *root* (:class:`str`; or :class:`pathlib.PurePath`) is the root
        directory to search for files.

        *on_error* (:class:`~collections.abc.Callable` or :data:`None`)
        optionally is the error handler for file-system exceptions. See
        :func:`~pathspec.util.iter_tree_files` for more information.

        *follow_links* (:class:`bool` or :data:`None`) optionally is whether
        to walk symbolic links that resolve to directories. See
        :func:`~pathspec.util.iter_tree_files` for more information.

        Returns the matched files (:class:`~collections.abc.Iterable` of
        :class:`str`).
        """
        ...
    match_tree = ...
