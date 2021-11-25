

import os

from .pattern import Pattern

"""
This module provides utility methods for dealing with path-specs.
"""
NORMALIZE_PATH_SEPS = ...
_registered_patterns = ...

def detailed_match_files(
    patterns: Iterable[Pattern],
    files: Iterable[Text],
    all_matches: Optional[bool] = ...,
) -> Dict[Text, MatchDetail]:
    """
    Matches the files to the patterns, and returns which patterns matched
    the files.

    *patterns* (:class:`~collections.abc.Iterable` of :class:`~pathspec.pattern.Pattern`)
    contains the patterns to use.

    *files* (:class:`~collections.abc.Iterable` of :class:`str`) contains
    the normalized file paths to be matched against *patterns*.

    *all_matches* (:class:`boot` or :data:`None`) is whether to return all
    matches patterns (:data:`True`), or only the last matched pattern
    (:data:`False`). Default is :data:`None` for :data:`False`.

    Returns the matched files (:class:`dict`) which maps each matched file
    (:class:`str`) to the patterns that matched in order (:class:`.MatchDetail`).
    """
    ...

def iter_tree_entries(
    root: Text, on_error: Optional[Callable] = ..., follow_links: Optional[bool] = ...
) -> Iterator[TreeEntry]:
    """
    Walks the specified directory for all files and directories.

    *root* (:class:`str`) is the root directory to search.

    *on_error* (:class:`~collections.abc.Callable` or :data:`None`)
    optionally is the error handler for file-system exceptions. It will be
    called with the exception (:exc:`OSError`). Reraise the exception to
    abort the walk. Default is :data:`None` to ignore file-system
    exceptions.

    *follow_links* (:class:`bool` or :data:`None`) optionally is whether
    to walk symbolic links that resolve to directories. Default is
    :data:`None` for :data:`True`.

    Raises :exc:`RecursionError` if recursion is detected.

    Returns an :class:`~collections.abc.Iterator` yielding each file or
    directory entry (:class:`.TreeEntry`) relative to *root*.
    """
    ...

def iter_tree_files(
    root: Text, on_error: Optional[Callable] = ..., follow_links: Optional[bool] = ...
) -> Iterator[Text]:
    """
    Walks the specified directory for all files.

    *root* (:class:`str`) is the root directory to search for files.

    *on_error* (:class:`~collections.abc.Callable` or :data:`None`)
    optionally is the error handler for file-system exceptions. It will be
    called with the exception (:exc:`OSError`). Reraise the exception to
    abort the walk. Default is :data:`None` to ignore file-system
    exceptions.

    *follow_links* (:class:`bool` or :data:`None`) optionally is whether
    to walk symbolic links that resolve to directories. Default is
    :data:`None` for :data:`True`.

    Raises :exc:`RecursionError` if recursion is detected.

    Returns an :class:`~collections.abc.Iterator` yielding the path to
    each file (:class:`str`) relative to *root*.
    """
    ...

iter_tree = ...

def lookup_pattern(name: Text) -> Callable[[AnyStr], Pattern]:
    """
    Lookups a registered pattern factory by name.

    *name* (:class:`str`) is the name of the pattern factory.

    Returns the registered pattern factory (:class:`~collections.abc.Callable`).
    If no pattern factory is registered, raises :exc:`KeyError`.
    """
    ...

def match_file(patterns: Iterable[Pattern], file: Text) -> bool:
    """
    Matches the file to the patterns.

    *patterns* (:class:`~collections.abc.Iterable` of :class:`~pathspec.pattern.Pattern`)
    contains the patterns to use.

    *file* (:class:`str`) is the normalized file path to be matched
    against *patterns*.

    Returns :data:`True` if *file* matched; otherwise, :data:`False`.
    """
    ...

def match_files(patterns: Iterable[Pattern], files: Iterable[Text]) -> Set[Text]:
    """
    Matches the files to the patterns.

    *patterns* (:class:`~collections.abc.Iterable` of :class:`~pathspec.pattern.Pattern`)
    contains the patterns to use.

    *files* (:class:`~collections.abc.Iterable` of :class:`str`) contains
    the normalized file paths to be matched against *patterns*.

    Returns the matched files (:class:`set` of :class:`str`).
    """
    ...

def normalize_file(
    file: Union[Text, PathLike], separators: Optional[Collection[Text]] = ...
) -> Text:
    """
    Normalizes the file path to use the POSIX path separator (i.e.,
    ``'/'``), and make the paths relative (remove leading ``'/'``).

    *file* (:class:`str` or :class:`pathlib.PurePath`) is the file path.

    *separators* (:class:`~collections.abc.Collection` of :class:`str`; or
    :data:`None`) optionally contains the path separators to normalize.
    This does not need to include the POSIX path separator (``'/'``), but
    including it will not affect the results. Default is :data:`None` for
    :data:`NORMALIZE_PATH_SEPS`. To prevent normalization, pass an empty
    container (e.g., an empty tuple ``()``).

    Returns the normalized file path (:class:`str`).
    """
    ...

def normalize_files(
    files: Iterable[Union[str, PathLike]], separators: Optional[Collection[Text]] = ...
) -> Dict[Text, List[Union[str, PathLike]]]:
    """
    Normalizes the file paths to use the POSIX path separator.

    *files* (:class:`~collections.abc.Iterable` of :class:`str` or
    :class:`pathlib.PurePath`) contains the file paths to be normalized.

    *separators* (:class:`~collections.abc.Collection` of :class:`str`; or
    :data:`None`) optionally contains the path separators to normalize.
    See :func:`normalize_file` for more information.

    Returns a :class:`dict` mapping the each normalized file path
    (:class:`str`) to the original file paths (:class:`list` of
    :class:`str` or :class:`pathlib.PurePath`).
    """
    ...

def register_pattern(
    name: Text,
    pattern_factory: Callable[[AnyStr], Pattern],
    override: Optional[bool] = ...,
) -> None:
    """
    Registers the specified pattern factory.

    *name* (:class:`str`) is the name to register the pattern factory
    under.

    *pattern_factory* (:class:`~collections.abc.Callable`) is used to
    compile patterns. It must accept an uncompiled pattern (:class:`str`)
    and return the compiled pattern (:class:`.Pattern`).

    *override* (:class:`bool` or :data:`None`) optionally is whether to
    allow overriding an already registered pattern under the same name
    (:data:`True`), instead of raising an :exc:`AlreadyRegisteredError`
    (:data:`False`). Default is :data:`None` for :data:`False`.
    """
    ...

class AlreadyRegisteredError(Exception):
    """
    The :exc:`AlreadyRegisteredError` exception is raised when a pattern
    factory is registered under a name already in use.
    """

    def __init__(
        self, name: Text, pattern_factory: Callable[[AnyStr], Pattern]
    ) -> None:
        """
        Initializes the :exc:`AlreadyRegisteredError` instance.

        *name* (:class:`str`) is the name of the registered pattern.

        *pattern_factory* (:class:`~collections.abc.Callable`) is the
        registered pattern factory.
        """
        ...
    @property
    def message(self) -> Text:
        """
        *message* (:class:`str`) is the error message.
        """
        ...
    @property
    def name(self) -> Text:
        """
        *name* (:class:`str`) is the name of the registered pattern.
        """
        ...
    @property
    def pattern_factory(self) -> Callable[[AnyStr], Pattern]:
        """
        *pattern_factory* (:class:`~collections.abc.Callable`) is the
        registered pattern factory.
        """
        ...

class RecursionError(Exception):
    """
    The :exc:`RecursionError` exception is raised when recursion is
    detected.
    """

    def __init__(self, real_path: Text, first_path: Text, second_path: Text) -> None:
        """
        Initializes the :exc:`RecursionError` instance.

        *real_path* (:class:`str`) is the real path that recursion was
        encountered on.

        *first_path* (:class:`str`) is the first path encountered for
        *real_path*.

        *second_path* (:class:`str`) is the second path encountered for
        *real_path*.
        """
        ...
    @property
    def first_path(self) -> Text:
        """
        *first_path* (:class:`str`) is the first path encountered for
        :attr:`self.real_path <RecursionError.real_path>`.
        """
        ...
    @property
    def message(self) -> Text:
        """
        *message* (:class:`str`) is the error message.
        """
        ...
    @property
    def real_path(self) -> Text:
        """
        *real_path* (:class:`str`) is the real path that recursion was
        encountered on.
        """
        ...
    @property
    def second_path(self) -> Text:
        """
        *second_path* (:class:`str`) is the second path encountered for
        :attr:`self.real_path <RecursionError.real_path>`.
        """
        ...

class MatchDetail:
    """
    The :class:`.MatchDetail` class contains information about
    """

    __slots__ = ...
    def __init__(self, patterns: Sequence[Pattern]) -> None:
        """
        Initialize the :class:`.MatchDetail` instance.

        *patterns* (:class:`~collections.abc.Sequence` of :class:`~pathspec.pattern.Pattern`)
        contains the patterns that matched the file in the order they were
        encountered.
        """
        ...

class TreeEntry:
    """
    The :class:`.TreeEntry` class contains information about a file-system
    entry.
    """

    __slots__ = ...
    def __init__(
        self, name: Text, path: Text, lstat: os.stat_result, stat: os.stat_result
    ) -> None:
        """
        Initialize the :class:`.TreeEntry` instance.

        *name* (:class:`str`) is the base name of the entry.

        *path* (:class:`str`) is the relative path of the entry.

        *lstat* (:class:`~os.stat_result`) is the stat result of the direct
        entry.

        *stat* (:class:`~os.stat_result`) is the stat result of the entry,
        potentially linked.
        """
        ...
    def is_dir(self, follow_links: Optional[bool] = ...) -> bool:
        """
        Get whether the entry is a directory.

        *follow_links* (:class:`bool` or :data:`None`) is whether to follow
        symbolic links. If this is :data:`True`, a symlink to a directory
        will result in :data:`True`. Default is :data:`None` for :data:`True`.

        Returns whether the entry is a directory (:class:`bool`).
        """
        ...
    def is_file(self, follow_links: Optional[bool] = ...) -> bool:
        """
        Get whether the entry is a regular file.

        *follow_links* (:class:`bool` or :data:`None`) is whether to follow
        symbolic links. If this is :data:`True`, a symlink to a regular file
        will result in :data:`True`. Default is :data:`None` for :data:`True`.

        Returns whether the entry is a regular file (:class:`bool`).
        """
        ...
    def is_symlink(self) -> bool:
        """
        Returns whether the entry is a symbolic link (:class:`bool`).
        """
        ...
    def stat(self, follow_links: Optional[bool] = ...) -> os.stat_result:
        """
        Get the cached stat result for the entry.

        *follow_links* (:class:`bool` or :data:`None`) is whether to follow
        symbolic links. If this is :data:`True`, the stat result of the
        linked file will be returned. Default is :data:`None` for :data:`True`.

        Returns that stat result (:class:`~os.stat_result`).
        """
        ...
