

import abc
import typing
from datetime import datetime
from threading import RLock
from types import TracebackType
from typing import (
    IO,
    Any,
    AnyStr,
    BinaryIO,
    Callable,
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Text,
    Tuple,
    Type,
    Union,
)

import six

from .enums import ResourceType
from .glob import BoundGlobber
from .info import Info, RawInfo
from .permissions import Permissions
from .subfs import SubFS
from .walk import BoundWalker, Walker

"""PyFilesystem base class.

The filesystem base class is common to all filesystems. If you
familiarize yourself with this (rather straightforward) API, you
can work with any of the supported filesystems.

"""

_F = typing.TypeVar("_F", bound="FS")
_T = typing.TypeVar("_T", bound="FS")
_OpendirFactory = Callable[[_T, Text], SubFS[_T]]
__all__ = ["FS"]

@six.add_metaclass(abc.ABCMeta)
class FS:
    """Base class for FS objects."""

    _meta: Dict[Text, Union[Text, int, bool, None]] = ...
    walker_class = Walker
    subfs_class = None
    def __init__(self) -> None:
        """Create a filesystem. See help(type(self)) for accurate signature."""
        ...
    def __del__(self) -> None:
        """Auto-close the filesystem on exit."""
        ...
    def __enter__(self) -> FS:
        """Allow use of filesystem as a context manager."""
        ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Close filesystem on exit."""
        ...
    @property
    def glob(self) -> BoundGlobber:
        """`~fs.glob.BoundGlobber`: a globber object.."""
        ...
    @property
    def walk(self: _F) -> BoundWalker[_F]:
        """`~fs.walk.BoundWalker`: a walker bound to this filesystem."""
        ...
    @abc.abstractmethod
    def getinfo(self, path: Text, namespaces: Optional[Collection[Text]] = ...) -> Info:
        """Get information about a resource on a filesystem.

        Arguments:
            path (str): A path to a resource on the filesystem.
            namespaces (list, optional): Info namespaces to query. The
                `"basic"` namespace is alway included in the returned
                info, whatever the value of `namespaces` may be.

        Returns:
            ~fs.info.Info: resource information object.

        Raises:
            fs.errors.ResourceNotFound: If ``path`` does not exist.

        For more information regarding resource information, see :ref:`info`.

        """
        ...
    @abc.abstractmethod
    def listdir(self, path: Text) -> List[Text]:
        """Get a list of the resource names in a directory.

        This method will return a list of the resources in a directory.
        A *resource* is a file, directory, or one of the other types
        defined in `~fs.enums.ResourceType`.

        Arguments:
            path (str): A path to a directory on the filesystem

        Returns:
            list: list of names, relative to ``path``.

        Raises:
            fs.errors.DirectoryExpected: If ``path`` is not a directory.
            fs.errors.ResourceNotFound: If ``path`` does not exist.

        """
        ...
    @abc.abstractmethod
    def makedir(
        self, path: Text, permissions: Optional[Permissions] = ..., recreate: bool = ...
    ) -> SubFS[FS]:
        """Make a directory.

        Arguments:
            path (str): Path to directory from root.
            permissions (~fs.permissions.Permissions, optional): a
                `Permissions` instance, or `None` to use default.
            recreate (bool): Set to `True` to avoid raising an error if
                the directory already exists (defaults to `False`).

        Returns:
            ~fs.subfs.SubFS: a filesystem whose root is the new directory.

        Raises:
            fs.errors.DirectoryExists: If the path already exists.
            fs.errors.ResourceNotFound: If the path is not found.

        """
        ...
    @abc.abstractmethod
    def openbin(
        self, path: Text, mode: Text = ..., buffering: int = ..., **options: Any
    ) -> BinaryIO:
        """Open a binary file-like object.

        Arguments:
            path (str): A path on the filesystem.
            mode (str): Mode to open file (must be a valid non-text mode,
                defaults to *r*). Since this method only opens binary files,
                the ``b`` in the mode string is implied.
            buffering (int): Buffering policy (-1 to use default buffering,
                0 to disable buffering, or any positive integer to indicate
                a buffer size).
            **options: keyword arguments for any additional information
                required by the filesystem (if any).

        Returns:
            io.IOBase: a *file-like* object.

        Raises:
            fs.errors.FileExpected: If ``path`` exists and is not a file.
            fs.errors.FileExists: If the ``path`` exists, and
                *exclusive mode* is specified (``x`` in the mode).
            fs.errors.ResourceNotFound: If ``path`` does not exist and
                ``mode`` does not imply creating the file, or if any
                ancestor of ``path`` does not exist.

        """
        ...
    @abc.abstractmethod
    def remove(self, path: Text) -> None:
        """Remove a file from the filesystem.

        Arguments:
            path (str): Path of the file to remove.

        Raises:
            fs.errors.FileExpected: If the path is a directory.
            fs.errors.ResourceNotFound: If the path does not exist.

        """
        ...
    @abc.abstractmethod
    def removedir(self, path: Text) -> None:
        """Remove a directory from the filesystem.

        Arguments:
            path (str): Path of the directory to remove.

        Raises:
            fs.errors.DirectoryNotEmpty: If the directory is not empty (
                see `~fs.base.FS.removetree` for a way to remove the
                directory contents).
            fs.errors.DirectoryExpected: If the path does not refer to
                a directory.
            fs.errors.ResourceNotFound: If no resource exists at the
                given path.
            fs.errors.RemoveRootError: If an attempt is made to remove
                the root directory (i.e. ``'/'``)

        """
        ...
    @abc.abstractmethod
    def setinfo(self, path: Text, info: RawInfo) -> None:
        """Set info on a resource.

        This method is the complement to `~fs.base.FS.getinfo`
        and is used to set info values on a resource.

        Arguments:
            path (str): Path to a resource on the filesystem.
            info (dict): Dictionary of resource info.

        Raises:
            fs.errors.ResourceNotFound: If ``path`` does not exist
                on the filesystem

        The ``info`` dict should be in the same format as the raw
        info returned by ``getinfo(file).raw``.

        Example:
            >>> details_info = {"details": {
            ...     "modified": time.time()
            ... }}
            >>> my_fs.setinfo('file.txt', details_info)

        """
        ...
    def appendbytes(self, path: Text, data: bytes) -> None:
        """Append bytes to the end of a file, creating it if needed.

        Arguments:
            path (str): Path to a file.
            data (bytes): Bytes to append.

        Raises:
            TypeError: If ``data`` is not a `bytes` instance.
            fs.errors.ResourceNotFound: If a parent directory of
                ``path`` does not exist.

        """
        ...
    def appendtext(
        self,
        path: Text,
        text: Text,
        encoding: Text = ...,
        errors: Optional[Text] = ...,
        newline: Text = ...,
    ) -> None:
        """Append text to the end of a file, creating it if needed.

        Arguments:
            path (str): Path to a file.
            text (str): Text to append.
            encoding (str): Encoding for text files (defaults to ``utf-8``).
            errors (str, optional): What to do with unicode decode errors
                (see `codecs` module for more information).
            newline (str): Newline parameter.

        Raises:
            TypeError: if ``text`` is not an unicode string.
            fs.errors.ResourceNotFound: if a parent directory of
                ``path`` does not exist.

        """
        ...
    def close(self) -> None:
        """Close the filesystem and release any resources.

        It is important to call this method when you have finished
        working with the filesystem. Some filesystems may not finalize
        changes until they are closed (archives for example). You may
        call this method explicitly (it is safe to call close multiple
        times), or you can use the filesystem as a context manager to
        automatically close.

        Example:
            >>> with OSFS('~/Desktop') as desktop_fs:
            ...    desktop_fs.writetext(
            ...        'note.txt',
            ...        "Don't forget to tape Game of Thrones"
            ...    )

        If you attempt to use a filesystem that has been closed, a
        `~fs.errors.FilesystemClosed` exception will be thrown.

        """
        ...
    def copy(self, src_path: Text, dst_path: Text, overwrite: bool = ...) -> None:
        """Copy file contents from ``src_path`` to ``dst_path``.

        Arguments:
            src_path (str): Path of source file.
            dst_path (str): Path to destination file.
            overwrite (bool): If `True`, overwrite the destination file
                if it exists (defaults to `False`).

        Raises:
            fs.errors.DestinationExists: If ``dst_path`` exists,
                and ``overwrite`` is `False`.
            fs.errors.ResourceNotFound: If a parent directory of
                ``dst_path`` does not exist.
            fs.errors.FileExpected: If ``src_path`` is not a file.

        """
        ...
    def copydir(self, src_path: Text, dst_path: Text, create: bool = ...) -> None:
        """Copy the contents of ``src_path`` to ``dst_path``.

        Arguments:
            src_path (str): Path of source directory.
            dst_path (str): Path to destination directory.
            create (bool): If `True`, then ``dst_path`` will be created
                if it doesn't exist already (defaults to `False`).

        Raises:
            fs.errors.ResourceNotFound: If the ``dst_path``
                does not exist, and ``create`` is not `True`.
            fs.errors.DirectoryExpected: If ``src_path`` is not a
                directory.

        """
        ...
    def create(self, path: Text, wipe: bool = ...) -> bool:
        """Create an empty file.

        The default behavior is to create a new file if one doesn't
        already exist. If ``wipe`` is `True`, any existing file will
        be truncated.

        Arguments:
            path (str): Path to a new file in the filesystem.
            wipe (bool): If `True`, truncate any existing
                file to 0 bytes (defaults to `False`).

        Returns:
            bool: `True` if a new file had to be created.

        """
        ...
    def desc(self, path: Text) -> Text:
        """Return a short descriptive text regarding a path.

        Arguments:
            path (str): A path to a resource on the filesystem.

        Returns:
            str: a short description of the path.

        Raises:
            fs.errors.ResourceNotFound: If ``path`` does not exist.

        """
        ...
    def exists(self, path: Text) -> bool:
        """Check if a path maps to a resource.

        Arguments:
            path (str): Path to a resource.

        Returns:
            bool: `True` if a resource exists at the given path.

        """
        ...
    def filterdir(
        self,
        path: Text,
        files: Optional[Iterable[Text]] = ...,
        dirs: Optional[Iterable[Text]] = ...,
        exclude_dirs: Optional[Iterable[Text]] = ...,
        exclude_files: Optional[Iterable[Text]] = ...,
        namespaces: Optional[Collection[Text]] = ...,
        page: Optional[Tuple[int, int]] = ...,
    ) -> Iterator[Info]:
        """Get an iterator of resource info, filtered by patterns.

        This method enhances `~fs.base.FS.scandir` with additional
        filtering functionality.

        Arguments:
            path (str): A path to a directory on the filesystem.
            files (list, optional): A list of UNIX shell-style patterns
                to filter file names, e.g. ``['*.py']``.
            dirs (list, optional): A list of UNIX shell-style patterns
                to filter directory names.
            exclude_dirs (list, optional): A list of patterns used
                to exclude directories.
            exclude_files (list, optional): A list of patterns used
                to exclude files.
            namespaces (list, optional): A list of namespaces to include
                in the resource information, e.g. ``['basic', 'access']``.
            page (tuple, optional): May be a tuple of ``(<start>, <end>)``
                indexes to return an iterator of a subset of the resource
                info, or `None` to iterate over the entire directory.
                Paging a directory scan may be necessary for very large
                directories.

        Returns:
            ~collections.abc.Iterator: an iterator of `Info` objects.

        """
        ...
    def readbytes(self, path: Text) -> bytes:
        """Get the contents of a file as bytes.

        Arguments:
            path (str): A path to a readable file on the filesystem.

        Returns:
            bytes: the file contents.

        Raises:
            fs.errors.FileExpected: if ``path`` exists but is not a file.
            fs.errors.ResourceNotFound: if ``path`` does not exist.

        """
        ...
    getbytes = readbytes
    def download(
        self,
        path: Text,
        file: BinaryIO,
        chunk_size: Optional[int] = ...,
        **options: Any
    ) -> None:
        """Copy a file from the filesystem to a file-like object.

        This may be more efficient that opening and copying files
        manually if the filesystem supplies an optimized method.

        Note that the file object ``file`` will *not* be closed by this
        method. Take care to close it after this method completes
        (ideally with a context manager).

        Arguments:
            path (str): Path to a resource.
            file (file-like): A file-like object open for writing in
                binary mode.
            chunk_size (int, optional): Number of bytes to read at a
                time, if a simple copy is used, or `None` to use
                sensible default.
            **options: Implementation specific options required to open
                the source file.

        Example:
            >>> with open('starwars.mov', 'wb') as write_file:
            ...     my_fs.download('/Videos/starwars.mov', write_file)

        Raises:
            fs.errors.ResourceNotFound: if ``path`` does not exist.

        """
        ...
    getfile = download
    def readtext(
        self,
        path: Text,
        encoding: Optional[Text] = ...,
        errors: Optional[Text] = ...,
        newline: Text = ...,
    ) -> Text:
        """Get the contents of a file as a string.

        Arguments:
            path (str): A path to a readable file on the filesystem.
            encoding (str, optional): Encoding to use when reading contents
                in text mode (defaults to `None`, reading in binary mode).
            errors (str, optional): Unicode errors parameter.
            newline (str): Newlines parameter.

        Returns:
            str: file contents.

        Raises:
            fs.errors.ResourceNotFound: If ``path`` does not exist.

        """
        ...
    gettext = readtext
    def getmeta(self, namespace: Text = ...) -> Mapping[Text, object]:
        """Get meta information regarding a filesystem.

        Arguments:
            namespace (str): The meta namespace (defaults
                to ``"standard"``).

        Returns:
            dict: the meta information.

        Meta information is associated with a *namespace* which may be
        specified with the ``namespace`` parameter. The default namespace,
        ``"standard"``, contains common information regarding the
        filesystem's capabilities. Some filesystems may provide other
        namespaces which expose less common or implementation specific
        information. If a requested namespace is not supported by
        a filesystem, then an empty dictionary will be returned.

        The ``"standard"`` namespace supports the following keys:

        =================== ============================================
        key                 Description
        ------------------- --------------------------------------------
        case_insensitive    `True` if this filesystem is case
                            insensitive.
        invalid_path_chars  A string containing the characters that
                            may not be used on this filesystem.
        max_path_length     Maximum number of characters permitted in
                            a path, or `None` for no limit.
        max_sys_path_length Maximum number of characters permitted in
                            a sys path, or `None` for no limit.
        network             `True` if this filesystem requires a
                            network.
        read_only           `True` if this filesystem is read only.
        supports_rename     `True` if this filesystem supports an
                            `os.rename` operation.
        =================== ============================================

        Most builtin filesystems will provide all these keys, and third-
        party filesystems should do so whenever possible, but a key may
        not be present if there is no way to know the value.

        Note:
            Meta information is constant for the lifetime of the
            filesystem, and may be cached.

        """
        ...
    def getsize(self, path: Text) -> int:
        """Get the size (in bytes) of a resource.

        Arguments:
            path (str): A path to a resource.

        Returns:
            int: the *size* of the resource.

        Raises:
            fs.errors.ResourceNotFound: if ``path`` does not exist.

        The *size* of a file is the total number of readable bytes,
        which may not reflect the exact number of bytes of reserved
        disk space (or other storage medium).

        The size of a directory is the number of bytes of overhead
        use to store the directory entry.

        """
        ...
    def getsyspath(self, path: Text) -> Text:
        """Get the *system path* of a resource.

        Arguments:
            path (str): A path on the filesystem.

        Returns:
            str: the *system path* of the resource, if any.

        Raises:
            fs.errors.NoSysPath: If there is no corresponding system path.

        A system path is one recognized by the OS, that may be used
        outside of PyFilesystem (in an application or a shell for
        example). This method will get the corresponding system path
        that would be referenced by ``path``.

        Not all filesystems have associated system paths. Network and
        memory based filesystems, for example, may not physically store
        data anywhere the OS knows about. It is also possible for some
        paths to have a system path, whereas others don't.

        This method will always return a str on Py3.* and unicode
        on Py2.7. See `~getospath` if you need to encode the path as
        bytes.

        If ``path`` doesn't have a system path, a `~fs.errors.NoSysPath`
        exception will be thrown.

        Note:
            A filesystem may return a system path even if no
            resource is referenced by that path -- as long as it can
            be certain what that system path would be.

        """
        ...
    def getospath(self, path: Text) -> bytes:
        """Get the *system path* to a resource, in the OS' prefered encoding.

        Arguments:
            path (str): A path on the filesystem.

        Returns:
            str: the *system path* of the resource, if any.

        Raises:
            fs.errors.NoSysPath: If there is no corresponding system path.

        This method takes the output of `~getsyspath` and encodes it to
        the filesystem's prefered encoding. In Python3 this step is
        not required, as the `os` module will do it automatically. In
        Python2.7, the encoding step is required to support filenames
        on the filesystem that don't encode correctly.

        Note:
            If you want your code to work in Python2.7 and Python3 then
            use this method if you want to work with the OS filesystem
            outside of the OSFS interface.

        """
        ...
    def gettype(self, path: Text) -> ResourceType:
        """Get the type of a resource.

        Arguments:
            path (str): A path on the filesystem.

        Returns:
            ~fs.enums.ResourceType: the type of the resource.

        Raises:
            fs.errors.ResourceNotFound: if ``path`` does not exist.

        A type of a resource is an integer that identifies the what
        the resource references. The standard type integers may be one
        of the values in the `~fs.enums.ResourceType` enumerations.

        The most common resource types, supported by virtually all
        filesystems are ``directory`` (1) and ``file`` (2), but the
        following types are also possible:

        ===================   ======
        ResourceType          value
        -------------------   ------
        unknown               0
        directory             1
        file                  2
        character             3
        block_special_file    4
        fifo                  5
        socket                6
        symlink               7
        ===================   ======

        Standard resource types are positive integers, negative values
        are reserved for implementation specific resource types.

        """
        ...
    def geturl(self, path: Text, purpose: Text = ...) -> Text:
        """Get the URL to a given resource.

        Arguments:
            path (str): A path on the filesystem
            purpose (str): A short string that indicates which URL
                to retrieve for the given path (if there is more than
                one). The default is ``'download'``, which should return
                a URL that serves the file. Other filesystems may support
                other values for ``purpose``.

        Returns:
            str: a URL.

        Raises:
            fs.errors.NoURL: If the path does not map to a URL.

        """
        ...
    def hassyspath(self, path: Text) -> bool:
        """Check if a path maps to a system path.

        Arguments:
            path (str): A path on the filesystem.

        Returns:
            bool: `True` if the resource at ``path`` has a *syspath*.

        """
        ...
    def hasurl(self, path: Text, purpose: Text = ...) -> bool:
        """Check if a path has a corresponding URL.

        Arguments:
            path (str): A path on the filesystem.
            purpose (str): A purpose parameter, as given in
                `~fs.base.FS.geturl`.

        Returns:
            bool: `True` if an URL for the given purpose exists.

        """
        ...
    def isclosed(self) -> bool:
        """Check if the filesystem is closed."""
        ...
    def isdir(self, path: Text) -> bool:
        """Check if a path maps to an existing directory.

        Arguments:
            path (str): A path on the filesystem.

        Returns:
            bool: `True` if ``path`` maps to a directory.

        """
        ...
    def isempty(self, path: Text) -> bool:
        """Check if a directory is empty.

        A directory is considered empty when it does not contain
        any file or any directory.

        Arguments:
            path (str): A path to a directory on the filesystem.

        Returns:
            bool: `True` if the directory is empty.

        Raises:
            errors.DirectoryExpected: If ``path`` is not a directory.
            errors.ResourceNotFound: If ``path`` does not exist.

        """
        ...
    def isfile(self, path: Text) -> bool:
        """Check if a path maps to an existing file.

        Arguments:
            path (str): A path on the filesystem.

        Returns:
            bool: `True` if ``path`` maps to a file.

        """
        ...
    def islink(self, path: Text) -> bool:
        """Check if a path maps to a symlink.

        Arguments:
            path (str): A path on the filesystem.

        Returns:
            bool: `True` if ``path`` maps to a symlink.

        """
        ...
    def lock(self) -> RLock:
        """Get a context manager that *locks* the filesystem.

        Locking a filesystem gives a thread exclusive access to it.
        Other threads will block until the threads with the lock has
        left the context manager.

        Returns:
            threading.RLock: a lock specific to the filesystem instance.

        Example:
            >>> with my_fs.lock():  # May block
            ...    # code here has exclusive access to the filesystem
            ...    pass

        It is a good idea to put a lock around any operations that you
        would like to be *atomic*. For instance if you are copying
        files, and you don't want another thread to delete or modify
        anything while the copy is in progress.

        Locking with this method is only required for code that calls
        multiple filesystem methods. Individual methods are thread safe
        already, and don't need to be locked.

        Note:
            This only locks at the Python level. There is nothing to
            prevent other processes from modifying the filesystem
            outside of the filesystem instance.

        """
        ...
    def movedir(self, src_path: Text, dst_path: Text, create: bool = ...) -> None:
        """Move directory ``src_path`` to ``dst_path``.

        Arguments:
            src_path (str): Path of source directory on the filesystem.
            dst_path (str): Path to destination directory.
            create (bool): If `True`, then ``dst_path`` will be created
                if it doesn't exist already (defaults to `False`).

        Raises:
            fs.errors.ResourceNotFound: if ``dst_path`` does not exist,
                and ``create`` is `False`.
            fs.errors.DirectoryExpected: if ``src_path`` or one of its
                ancestors is not a directory.

        """
        ...
    def makedirs(
        self, path: Text, permissions: Optional[Permissions] = ..., recreate: bool = ...
    ) -> SubFS[FS]:
        """Make a directory, and any missing intermediate directories.

        Arguments:
            path (str): Path to directory from root.
            permissions (~fs.permissions.Permissions, optional): Initial
                permissions, or `None` to use defaults.
            recreate (bool):  If `False` (the default), attempting to
                create an existing directory will raise an error. Set
                to `True` to ignore existing directories.

        Returns:
            ~fs.subfs.SubFS: A sub-directory filesystem.

        Raises:
            fs.errors.DirectoryExists: if the path is already
                a directory, and ``recreate`` is `False`.
            fs.errors.DirectoryExpected: if one of the ancestors
                in the path is not a directory.

        """
        ...
    def move(self, src_path: Text, dst_path: Text, overwrite: bool = ...) -> None:
        """Move a file from ``src_path`` to ``dst_path``.

        Arguments:
            src_path (str): A path on the filesystem to move.
            dst_path (str): A path on the filesystem where the source
                file will be written to.
            overwrite (bool): If `True`, destination path will be
                overwritten if it exists.

        Raises:
            fs.errors.FileExpected: If ``src_path`` maps to a
                directory instead of a file.
            fs.errors.DestinationExists: If ``dst_path`` exists,
                and ``overwrite`` is `False`.
            fs.errors.ResourceNotFound: If a parent directory of
                ``dst_path`` does not exist.

        """
        ...
    def open(
        self,
        path: Text,
        mode: Text = ...,
        buffering: int = ...,
        encoding: Optional[Text] = ...,
        errors: Optional[Text] = ...,
        newline: Text = ...,
        **options: Any
    ) -> IO[Any]:
        """Open a file.

        Arguments:
            path (str): A path to a file on the filesystem.
            mode (str): Mode to open the file object with
                (defaults to *r*).
            buffering (int): Buffering policy (-1 to use
                default buffering, 0 to disable buffering, 1 to select
                line buffering, of any positive integer to indicate
                a buffer size).
            encoding (str): Encoding for text files (defaults to
                ``utf-8``)
            errors (str, optional): What to do with unicode decode errors
                (see `codecs` module for more information).
            newline (str): Newline parameter.
            **options: keyword arguments for any additional information
                required by the filesystem (if any).

        Returns:
            io.IOBase: a *file-like* object.

        Raises:
            fs.errors.FileExpected: If the path is not a file.
            fs.errors.FileExists: If the file exists, and *exclusive mode*
                is specified (``x`` in the mode).
            fs.errors.ResourceNotFound: If the path does not exist.

        """
        ...
    def opendir(
        self: _F, path: Text, factory: Optional[_OpendirFactory[_F]] = ...
    ) -> SubFS[FS]:
        """Get a filesystem object for a sub-directory.

        Arguments:
            path (str): Path to a directory on the filesystem.
            factory (callable, optional): A callable that when invoked
                with an FS instance and ``path`` will return a new FS object
                representing the sub-directory contents. If no ``factory``
                is supplied then `~fs.subfs_class` will be used.

        Returns:
            ~fs.subfs.SubFS: A filesystem representing a sub-directory.

        Raises:
            fs.errors.ResourceNotFound: If ``path`` does not exist.
            fs.errors.DirectoryExpected: If ``path`` is not a directory.

        """
        ...
    def removetree(self, dir_path: Text) -> None:
        """Recursively remove a directory and all its contents.

        This method is similar to `~fs.base.FS.removedir`, but will
        remove the contents of the directory if it is not empty.

        Arguments:
            dir_path (str): Path to a directory on the filesystem.

        Raises:
            fs.errors.ResourceNotFound: If ``dir_path`` does not exist.
            fs.errors.DirectoryExpected: If ``dir_path`` is not a directory.

        Caution:
            A filesystem should never delete its root folder, so
            ``FS.removetree("/")`` has different semantics: the
            contents of the root folder will be deleted, but the
            root will be untouched::

                >>> home_fs = fs.open_fs("~")
                >>> home_fs.removetree("/")
                >>> home_fs.exists("/")
                True
                >>> home_fs.isempty("/")
                True

            Combined with `~fs.base.FS.opendir`, this can be used
            to clear a directory without removing the directory
            itself::

                >>> home_fs = fs.open_fs("~")
                >>> home_fs.opendir("/Videos").removetree("/")
                >>> home_fs.exists("/Videos")
                True
                >>> home_fs.isempty("/Videos")
                True

        """
        ...
    def scandir(
        self,
        path: Text,
        namespaces: Optional[Collection[Text]] = ...,
        page: Optional[Tuple[int, int]] = ...,
    ) -> Iterator[Info]:
        """Get an iterator of resource info.

        Arguments:
            path (str): A path to a directory on the filesystem.
            namespaces (list, optional): A list of namespaces to include
                in the resource information, e.g. ``['basic', 'access']``.
            page (tuple, optional): May be a tuple of ``(<start>, <end>)``
                indexes to return an iterator of a subset of the resource
                info, or `None` to iterate over the entire directory.
                Paging a directory scan may be necessary for very large
                directories.

        Returns:
            ~collections.abc.Iterator: an iterator of `Info` objects.

        Raises:
            fs.errors.DirectoryExpected: If ``path`` is not a directory.
            fs.errors.ResourceNotFound: If ``path`` does not exist.

        """
        ...
    def writebytes(self, path: Text, contents: bytes) -> None:
        """Copy binary data to a file.

        Arguments:
            path (str): Destination path on the filesystem.
            contents (bytes): Data to be written.

        Raises:
            TypeError: if contents is not bytes.

        """
        ...
    setbytes = writebytes
    def upload(
        self,
        path: Text,
        file: BinaryIO,
        chunk_size: Optional[int] = ...,
        **options: Any
    ) -> None:
        """Set a file to the contents of a binary file object.

        This method copies bytes from an open binary file to a file on
        the filesystem. If the destination exists, it will first be
        truncated.

        Arguments:
            path (str): A path on the filesystem.
            file (io.IOBase): a file object open for reading in
                binary mode.
            chunk_size (int, optional): Number of bytes to read at a
                time, if a simple copy is used, or `None` to use
                sensible default.
            **options: Implementation specific options required to open
                the source file.

        Raises:
            fs.errors.ResourceNotFound: If a parent directory of
                ``path`` does not exist.

        Note that the file object ``file`` will *not* be closed by this
        method. Take care to close it after this method completes
        (ideally with a context manager).

        Example:
            >>> with open('~/movies/starwars.mov', 'rb') as read_file:
            ...     my_fs.upload('starwars.mov', read_file)

        """
        ...
    setbinfile = upload
    def writefile(
        self,
        path: Text,
        file: IO[AnyStr],
        encoding: Optional[Text] = ...,
        errors: Optional[Text] = ...,
        newline: Text = ...,
    ) -> None:
        """Set a file to the contents of a file object.

        Arguments:
            path (str): A path on the filesystem.
            file (io.IOBase): A file object open for reading.
            encoding (str, optional): Encoding of destination file,
                defaults to `None` for binary.
            errors (str, optional): How encoding errors should be treated
                (same as `io.open`).
            newline (str): Newline parameter (same as `io.open`).

        This method is similar to `~FS.upload`, in that it copies data from a
        file-like object to a resource on the filesystem, but unlike ``upload``,
        this method also supports creating files in text-mode (if the ``encoding``
        argument is supplied).

        Note that the file object ``file`` will *not* be closed by this
        method. Take care to close it after this method completes
        (ideally with a context manager).

        Example:
            >>> with open('myfile.txt') as read_file:
            ...     my_fs.writefile('myfile.txt', read_file)

        """
        ...
    setfile = writefile
    def settimes(
        self,
        path: Text,
        accessed: Optional[datetime] = ...,
        modified: Optional[datetime] = ...,
    ) -> None:
        """Set the accessed and modified time on a resource.

        Arguments:
            path: A path to a resource on the filesystem.
            accessed (datetime, optional): The accessed time, or
                `None` (the default) to use the current time.
            modified (datetime, optional): The modified time, or
                `None` (the default) to use the same time as the
                ``accessed`` parameter.

        """
        ...
    def writetext(
        self,
        path: Text,
        contents: Text,
        encoding: Text = ...,
        errors: Optional[Text] = ...,
        newline: Text = ...,
    ) -> None:
        """Create or replace a file with text.

        Arguments:
            path (str): Destination path on the filesystem.
            contents (str): Text to be written.
            encoding (str, optional): Encoding of destination file
                (defaults to ``'utf-8'``).
            errors (str, optional): How encoding errors should be treated
                (same as `io.open`).
            newline (str): Newline parameter (same as `io.open`).

        Raises:
            TypeError: if ``contents`` is not a unicode string.

        """
        ...
    settext = writetext
    def touch(self, path: Text) -> None:
        """Touch a file on the filesystem.

        Touching a file means creating a new file if ``path`` doesn't
        exist, or update accessed and modified times if the path does
        exist. This method is similar to the linux command of the same
        name.

        Arguments:
            path (str): A path to a file on the filesystem.

        """
        ...
    def validatepath(self, path: Text) -> Text:
        """Validate a path, returning a normalized absolute path on sucess.

        Many filesystems have restrictions on the format of paths they
        support. This method will check that ``path`` is valid on the
        underlaying storage mechanism and throw a
        `~fs.errors.InvalidPath` exception if it is not.

        Arguments:
            path (str): A path.

        Returns:
            str: A normalized, absolute path.

        Raises:
            fs.errors.InvalidPath: If the path is invalid.
            fs.errors.FilesystemClosed: if the filesystem is closed.
            fs.errors.InvalidCharsInPath: If the path contains
                invalid characters.

        """
        ...
    def getbasic(self, path: Text) -> Info:
        """Get the *basic* resource info.

        This method is shorthand for the following::

            fs.getinfo(path, namespaces=['basic'])

        Arguments:
            path (str): A path on the filesystem.

        Returns:
            ~fs.info.Info: Resource information object for ``path``.

        Note:
            .. deprecated:: 2.4.13
                Please use `~FS.getinfo` directly, which is
                required to always return the *basic* namespace.

        """
        ...
    def getdetails(self, path: Text) -> Info:
        """Get the *details* resource info.

        This method is shorthand for the following::

            fs.getinfo(path, namespaces=['details'])

        Arguments:
            path (str): A path on the filesystem.

        Returns:
            ~fs.info.Info: Resource information object for ``path``.

        """
        ...
    def check(self) -> None:
        """Check if a filesystem may be used.

        Raises:
            fs.errors.FilesystemClosed: if the filesystem is closed.

        """
        ...
    def match(self, patterns: Optional[Iterable[Text]], name: Text) -> bool:
        """Check if a name matches any of a list of wildcards.

        If a filesystem is case *insensitive* (such as Windows) then
        this method will perform a case insensitive match (i.e. ``*.py``
        will match the same names as ``*.PY``). Otherwise the match will
        be case sensitive (``*.py`` and ``*.PY`` will match different
        names).

        Arguments:
            patterns (list, optional): A list of patterns, e.g.
                ``['*.py']``, or `None` to match everything.
            name (str): A file or directory name (not a path)

        Returns:
            bool: `True` if ``name`` matches any of the patterns.

        Raises:
            TypeError: If ``patterns`` is a single string instead of
                a list (or `None`).

        Example:
            >>> my_fs.match(['*.py'], '__init__.py')
            True
            >>> my_fs.match(['*.jpg', '*.png'], 'foo.gif')
            False

        Note:
            If ``patterns`` is `None` (or ``['*']``), then this
            method will always return `True`.

        """
        ...
    def tree(self, **kwargs: Any) -> None:
        """Render a tree view of the filesystem to stdout or a file.

        The parameters are passed to :func:`~fs.tree.render`.

        Keyword Arguments:
            path (str): The path of the directory to start rendering
                from (defaults to root folder, i.e. ``'/'``).
            file (io.IOBase): An open file-like object to render the
                tree, or `None` for stdout.
            encoding (str): Unicode encoding, or `None` to
                auto-detect.
            max_levels (int): Maximum number of levels to
                display, or `None` for no maximum.
            with_color (bool): Enable terminal color output,
                or `None` to auto-detect terminal.
            dirs_first (bool): Show directories first.
            exclude (list): Option list of directory patterns
                to exclude from the tree render.
            filter (list): Optional list of files patterns to
                match in the tree render.

        """
        ...
    def hash(self, path: Text, name: Text) -> Text:
        """Get the hash of a file's contents.

        Arguments:
            path(str): A path on the filesystem.
            name(str):
                One of the algorithms supported by the `hashlib` module,
                e.g. `"md5"` or `"sha256"`.

        Returns:
            str: The hex digest of the hash.

        Raises:
            fs.errors.UnsupportedHash: If the requested hash is not supported.
            fs.errors.ResourceNotFound: If ``path`` does not exist.
            fs.errors.FileExpected: If ``path`` exists but is not a file.

        """
        ...
