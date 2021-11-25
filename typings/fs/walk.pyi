

import typing
from typing import (
    Any,
    Callable,
    Collection,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Text,
    Tuple,
    Type,
)

from .base import FS
from .info import Info

"""Machinery for walking a filesystem.

*Walking* a filesystem means recursively visiting a directory and
any sub-directories. It is a fairly common requirement for copying,
searching etc. See :ref:`walking` for details.
"""
if typing.TYPE_CHECKING:
    OnError = Callable[[Text, Exception], bool]

_F = typing.TypeVar("_F", bound="FS")

class Step(NamedTuple):
    path: Text
    dirs: List[Info]
    files: List[Info]

class Walker:
    """A walker object recursively lists directories in a filesystem."""

    def __init__(
        self,
        ignore_errors: bool = ...,
        on_error: Optional[OnError] = ...,
        search: Text = ...,
        filter: Optional[List[Text]] = ...,
        exclude: Optional[List[Text]] = ...,
        filter_dirs: Optional[List[Text]] = ...,
        exclude_dirs: Optional[List[Text]] = ...,
        max_depth: Optional[int] = ...,
    ) -> None:
        """Create a new `Walker` instance.

        Arguments:
            ignore_errors (bool): If `True`, any errors reading a
                directory will be ignored, otherwise exceptions will
                be raised.
            on_error (callable, optional): If ``ignore_errors`` is `False`,
                then this callable will be invoked for a path and the
                exception object. It should return `True` to ignore the error,
                or `False` to re-raise it.
            search (str): If ``"breadth"`` then the directory will be
                walked *top down*. Set to ``"depth"`` to walk *bottom up*.
            filter (list, optional): If supplied, this parameter should be
                a list of filename patterns, e.g. ``["*.py"]``. Files will
                only be returned if the final component matches one of the
                patterns.
            exclude (list, optional): If supplied, this parameter should be
                a list of filename patterns, e.g. ``["~*"]``. Files matching
                any of these patterns will be removed from the walk.
            filter_dirs (list, optional): A list of patterns that will be used
                to match directories paths. The walk will only open directories
                that match at least one of these patterns.
            exclude_dirs (list, optional): A list of patterns that will be
                used to filter out directories from the walk. e.g.
                ``['*.svn', '*.git']``.
            max_depth (int, optional): Maximum directory depth to walk.

        """
        ...
    @classmethod
    def bind(cls, fs: _F) -> BoundWalker[_F]:
        """Bind a `Walker` instance to a given filesystem.

        This *binds* in instance of the Walker to a given filesystem, so
        that you won't need to explicitly provide the filesystem as a
        parameter.

        Arguments:
            fs (FS): A filesystem object.

        Returns:
            ~fs.walk.BoundWalker: a bound walker.

        Examples:
            Use this method to explicitly bind a filesystem instance::

                >>> walker = Walker.bind(my_fs)
                >>> for path in walker.files(filter=['*.py']):
                ...     print(path)
                /foo.py
                /bar.py

            Unless you have written a customized walker class, you will
            be unlikely to need to call this explicitly, as filesystem
            objects already have a ``walk`` attribute which is a bound
            walker object::

                >>> for path in my_fs.walk.files(filter=['*.py']):
                ...     print(path)
                /foo.py
                /bar.py

        """
        ...
    def __repr__(self) -> Text: ...
    def check_open_dir(self, fs: FS, path: Text, info: Info) -> bool:
        """Check if a directory should be opened.

        Override to exclude directories from the walk.

        Arguments:
            fs (FS): A filesystem instance.
            path (str): Path to directory.
            info (Info): A resource info object for the directory.

        Returns:
            bool: `True` if the directory should be opened.

        """
        ...
    def check_scan_dir(self, fs: FS, path: Text, info: Info) -> bool:
        """Check if a directory should be scanned.

        Override to omit scanning of certain directories. If a directory
        is omitted, it will appear in the walk but its files and
        sub-directories will not.

        Arguments:
            fs (FS): A filesystem instance.
            path (str): Path to directory.
            info (Info): A resource info object for the directory.

        Returns:
            bool: `True` if the directory should be scanned.

        """
        ...
    def check_file(self, fs: FS, info: Info) -> bool:
        """Check if a filename should be included.

        Override to exclude files from the walk.

        Arguments:
            fs (FS): A filesystem instance.
            info (Info): A resource info object.

        Returns:
            bool: `True` if the file should be included.

        """
        ...
    def walk(
        self, fs: FS, path: Text = ..., namespaces: Optional[Collection[Text]] = ...
    ) -> Iterator[Step]:
        """Walk the directory structure of a filesystem.

        Arguments:
            fs (FS): A filesystem instance.
            path (str): A path to a directory on the filesystem.
            namespaces (list, optional): A list of additional namespaces
                to add to the `Info` objects.

        Returns:
            collections.Iterator: an iterator of `~fs.walk.Step` instances.

        The return value is an iterator of ``(<path>, <dirs>, <files>)``
        named tuples,  where ``<path>`` is an absolute path to a
        directory, and ``<dirs>`` and ``<files>`` are a list of
        `~fs.info.Info` objects for directories and files in ``<path>``.

        Example:
            >>> walker = Walker(filter=['*.py'])
            >>> for path, dirs, files in walker.walk(my_fs, namespaces=["details"]):
            ...    print("[{}]".format(path))
            ...    print("{} directories".format(len(dirs)))
            ...    total = sum(info.size for info in files)
            ...    print("{} bytes".format(total))
            [/]
            2 directories
            55 bytes
            ...

        """
        ...
    def files(self, fs: FS, path: Text = ...) -> Iterator[Text]:
        """Walk a filesystem, yielding absolute paths to files.

        Arguments:
            fs (FS): A filesystem instance.
            path (str): A path to a directory on the filesystem.

        Yields:
            str: absolute path to files on the filesystem found
            recursively within the given directory.

        """
        ...
    def dirs(self, fs: FS, path: Text = ...) -> Iterator[Text]:
        """Walk a filesystem, yielding absolute paths to directories.

        Arguments:
            fs (FS): A filesystem instance.
            path (str): A path to a directory on the filesystem.

        Yields:
            str: absolute path to directories on the filesystem found
            recursively within the given directory.

        """
        ...
    def info(
        self, fs: FS, path: Text = ..., namespaces: Optional[Collection[Text]] = ...
    ) -> Iterator[Tuple[Text, Info]]:
        """Walk a filesystem, yielding tuples of ``(<path>, <info>)``.

        Arguments:
            fs (FS): A filesystem instance.
            path (str): A path to a directory on the filesystem.
            namespaces (list, optional): A list of additional namespaces
                to add to the `Info` objects.

        Yields:
            (str, Info): a tuple of ``(<absolute path>, <resource info>)``.

        """
        ...

class BoundWalker(typing.Generic[_F]):
    """A class that binds a `Walker` instance to a `FS` instance.

    You will typically not need to create instances of this class
    explicitly. Filesystems have a `~FS.walk` property which returns a
    `BoundWalker` object.

    Example:
        >>> tmp_fs = fs.tempfs.TempFS()
        >>> tmp_fs.walk
        BoundWalker(TempFS())

    A `BoundWalker` is callable. Calling it is an alias for the
    `~fs.walk.BoundWalker.walk` method.

    """

    def __init__(self, fs: _F, walker_class: Type[Walker] = ...) -> None:
        """Create a new walker bound to the given filesystem.

        Arguments:
            fs (FS): A filesystem instance.
            walker_class (type): A `~fs.walk.WalkerBase`
                sub-class. The default uses `~fs.walk.Walker`.

        """
        ...
    def __repr__(self) -> Text: ...
    def walk(
        self,
        path: Text = ...,
        namespaces: Optional[Collection[Text]] = ...,
        **kwargs: Any
    ) -> Iterator[Step]:
        """Walk the directory structure of a filesystem.

        Arguments:
            path (str):
            namespaces (list, optional): A list of namespaces to include
                in the resource information, e.g. ``['basic', 'access']``
                (defaults to ``['basic']``).

        Keyword Arguments:
            ignore_errors (bool): If `True`, any errors reading a
                directory will be ignored, otherwise exceptions will be
                raised.
            on_error (callable): If ``ignore_errors`` is `False`, then
                this callable will be invoked with a path and the exception
                object. It should return `True` to ignore the error, or
                `False` to re-raise it.
            search (str): If ``'breadth'`` then the directory will be
                walked *top down*. Set to ``'depth'`` to walk *bottom up*.
            filter (list): If supplied, this parameter should be a list
                of file name patterns, e.g. ``['*.py']``. Files will only be
                returned if the final component matches one of the
                patterns.
            exclude (list, optional): If supplied, this parameter should be
                a list of filename patterns, e.g. ``['~*', '.*']``. Files matching
                any of these patterns will be removed from the walk.
            filter_dirs (list, optional): A list of patterns that will be used
                to match directories paths. The walk will only open directories
                that match at least one of these patterns.
            exclude_dirs (list): A list of patterns that will be used
                to filter out directories from the walk, e.g. ``['*.svn',
                '*.git']``.
            max_depth (int, optional): Maximum directory depth to walk.

        Returns:
            ~collections.Iterator: an iterator of ``(<path>, <dirs>, <files>)``
            named tuples,  where ``<path>`` is an absolute path to a
            directory, and ``<dirs>`` and ``<files>`` are a list of
            `~fs.info.Info` objects for directories and files in ``<path>``.

        Example:
            >>> walker = Walker(filter=['*.py'])
            >>> for path, dirs, files in walker.walk(my_fs, namespaces=['details']):
            ...     print("[{}]".format(path))
            ...     print("{} directories".format(len(dirs)))
            ...     total = sum(info.size for info in files)
            ...     print("{} bytes".format(total))
            [/]
            2 directories
            55 bytes
            ...

        This method invokes `Walker.walk` with bound `FS` object.

        """
        ...
    __call__ = walk
    def files(self, path: Text = ..., **kwargs: Any) -> Iterator[Text]:
        """Walk a filesystem, yielding absolute paths to files.

        Arguments:
            path (str): A path to a directory.

        Keyword Arguments:
            ignore_errors (bool): If `True`, any errors reading a
                directory will be ignored, otherwise exceptions will be
                raised.
            on_error (callable): If ``ignore_errors`` is `False`, then
                this callable will be invoked with a path and the exception
                object. It should return `True` to ignore the error, or
                `False` to re-raise it.
            search (str): If ``'breadth'`` then the directory will be
                walked *top down*. Set to ``'depth'`` to walk *bottom up*.
            filter (list): If supplied, this parameter should be a list
                of file name patterns, e.g. ``['*.py']``. Files will only be
                returned if the final component matches one of the
                patterns.
            exclude (list, optional): If supplied, this parameter should be
                a list of filename patterns, e.g. ``['~*', '.*']``. Files matching
                any of these patterns will be removed from the walk.
            filter_dirs (list, optional): A list of patterns that will be used
                to match directories paths. The walk will only open directories
                that match at least one of these patterns.
            exclude_dirs (list): A list of patterns that will be used
                to filter out directories from the walk, e.g. ``['*.svn',
                '*.git']``.
            max_depth (int, optional): Maximum directory depth to walk.

        Returns:
            ~collections.Iterator: An iterator over file paths (absolute
            from the filesystem root).

        This method invokes `Walker.files` with the bound `FS` object.

        """
        ...
    def dirs(self, path: Text = ..., **kwargs: Any) -> Iterator[Text]:
        """Walk a filesystem, yielding absolute paths to directories.

        Arguments:
            path (str): A path to a directory.

        Keyword Arguments:
            ignore_errors (bool): If `True`, any errors reading a
                directory will be ignored, otherwise exceptions will be
                raised.
            on_error (callable): If ``ignore_errors`` is `False`, then
                this callable will be invoked with a path and the exception
                object. It should return `True` to ignore the error, or
                `False` to re-raise it.
            search (str): If ``'breadth'`` then the directory will be
                walked *top down*. Set to ``'depth'`` to walk *bottom up*.
            filter_dirs (list, optional): A list of patterns that will be used
                to match directories paths. The walk will only open directories
                that match at least one of these patterns.
            exclude_dirs (list): A list of patterns that will be used
                to filter out directories from the walk, e.g. ``['*.svn',
                '*.git']``.
            max_depth (int, optional): Maximum directory depth to walk.

        Returns:
            ~collections.Iterator: an iterator over directory paths
            (absolute from the filesystem root).

        This method invokes `Walker.dirs` with the bound `FS` object.

        """
        ...
    def info(
        self,
        path: Text = ...,
        namespaces: Optional[Collection[Text]] = ...,
        **kwargs: Any
    ) -> Iterator[Tuple[Text, Info]]:
        """Walk a filesystem, yielding path and `Info` of resources.

        Arguments:
            path (str): A path to a directory.
            namespaces (list, optional): A list of namespaces to include
                in the resource information, e.g. ``['basic', 'access']``
                (defaults to ``['basic']``).

        Keyword Arguments:
            ignore_errors (bool): If `True`, any errors reading a
                directory will be ignored, otherwise exceptions will be
                raised.
            on_error (callable): If ``ignore_errors`` is `False`, then
                this callable will be invoked with a path and the exception
                object. It should return `True` to ignore the error, or
                `False` to re-raise it.
            search (str): If ``'breadth'`` then the directory will be
                walked *top down*. Set to ``'depth'`` to walk *bottom up*.
            filter (list): If supplied, this parameter should be a list
                of file name patterns, e.g. ``['*.py']``. Files will only be
                returned if the final component matches one of the
                patterns.
            exclude (list, optional): If supplied, this parameter should be
                a list of filename patterns, e.g. ``['~*', '.*']``. Files matching
                any of these patterns will be removed from the walk.
            filter_dirs (list, optional): A list of patterns that will be used
                to match directories paths. The walk will only open directories
                that match at least one of these patterns.
            exclude_dirs (list): A list of patterns that will be used
                to filter out directories from the walk, e.g. ``['*.svn',
                '*.git']``.
            max_depth (int, optional): Maximum directory depth to walk.

        Returns:
            ~collections.Iterable: an iterable yielding tuples of
            ``(<absolute path>, <resource info>)``.

        This method invokes `Walker.info` with the bound `FS` object.

        """
        ...

default_walker = Walker()
walk = default_walker.walk
walk_files = default_walker.files
walk_info = default_walker.info
walk_dirs = default_walker.dirs
