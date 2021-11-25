

import contextlib
import typing
from typing import Callable, Iterator, List, Text, Tuple, Type, Union

from ..base import FS
from .base import Opener

"""`Registry` class mapping protocols and FS URLs to their `Opener`.
"""
if typing.TYPE_CHECKING: ...

class Registry:
    """A registry for `Opener` instances."""

    def __init__(self, default_opener: Text = ..., load_extern: bool = ...) -> None:
        """Create a registry object.

        Arguments:
            default_opener (str, optional): The protocol to use, if one
                is not supplied. The default is to use 'osfs', so that the
                FS URL is treated as a system path if no protocol is given.
            load_extern (bool, optional): Set to `True` to load openers from
                PyFilesystem2 extensions. Defaults to `False`.

        """
        ...
    def __repr__(self) -> Text: ...
    def install(
        self, opener: Union[Type[Opener], Opener, Callable[[], Opener]]
    ) -> Opener:
        """Install an opener.

        Arguments:
            opener (`Opener`): an `Opener` instance, or a callable that
                returns an opener instance.

        Note:
            May be used as a class decorator. For example::

                registry = Registry()
                @registry.install
                class ArchiveOpener(Opener):
                    protocols = ['zip', 'tar']

        """
        ...
    @property
    def protocols(self) -> List[Text]:
        """`list`: the list of supported protocols."""
        ...
    def get_opener(self, protocol: Text) -> Opener:
        """Get the opener class associated to a given protocol.

        Arguments:
            protocol (str): A filesystem protocol.

        Returns:
            Opener: an opener instance.

        Raises:
            ~fs.opener.errors.UnsupportedProtocol: If no opener
                could be found for the given protocol.
            EntryPointLoadingError: If the returned entry point
                is not an `Opener` subclass or could not be loaded
                successfully.

        """
        ...
    def open(
        self,
        fs_url: Text,
        writeable: bool = ...,
        create: bool = ...,
        cwd: Text = ...,
        default_protocol: Text = ...,
    ) -> Tuple[FS, Text]:
        """Open a filesystem from a FS URL.

        Returns a tuple of a filesystem object and a path. If there is
        no path in the FS URL, the path value will be `None`.

        Arguments:
            fs_url (str): A filesystem URL.
            writeable (bool, optional): `True` if the filesystem must be
                writeable.
            create (bool, optional): `True` if the filesystem should be
                created if it does not exist.
            cwd (str): The current working directory.

        Returns:
            (FS, str): a tuple of ``(<filesystem>, <path from url>)``

        """
        ...
    def open_fs(
        self,
        fs_url: Union[FS, Text],
        writeable: bool = ...,
        create: bool = ...,
        cwd: Text = ...,
        default_protocol: Text = ...,
    ) -> FS:
        """Open a filesystem from a FS URL (ignoring the path component).

        Arguments:
            fs_url (str): A filesystem URL. If a filesystem instance is
                given instead, it will be returned transparently.
            writeable (bool, optional): `True` if the filesystem must
                be writeable.
            create (bool, optional): `True` if the filesystem should be
                created if it does not exist.
            cwd (str): The current working directory (generally only
                relevant for OS filesystems).
            default_protocol (str): The protocol to use if one is not
                supplied in the FS URL (defaults to ``"osfs"``).

        Returns:
            ~fs.base.FS: A filesystem instance.

        Caution:
            The ``writeable`` parameter only controls whether the
            filesystem *needs* to be writable, which is relevant for
            some archive filesystems. Passing ``writeable=False`` will
            **not** make the return filesystem read-only. For this,
            consider using `fs.wrap.read_only` to wrap the returned
            instance.

        """
        ...
    @contextlib.contextmanager
    def manage_fs(
        self,
        fs_url: Union[FS, Text],
        create: bool = ...,
        writeable: bool = ...,
        cwd: Text = ...,
    ) -> Iterator[FS]:
        """Get a context manager to open and close a filesystem.

        Arguments:
            fs_url (FS or str): A filesystem instance or a FS URL.
            create (bool, optional): If `True`, then create the filesystem if
                it doesn't already exist.
            writeable (bool, optional): If `True`, then the filesystem
                must be writeable.
            cwd (str): The current working directory, if opening a
                `~fs.osfs.OSFS`.

        Sometimes it is convenient to be able to pass either a FS object
        *or* an FS URL to a function. This context manager handles the
        required logic for that.

        Example:
            The `~Registry.manage_fs` method can be used to define a small
            utility function::

                >>> def print_ls(list_fs):
                ...     '''List a directory.'''
                ...     with manage_fs(list_fs) as fs:
                ...         print(' '.join(fs.listdir()))

            This function may be used in two ways. You may either pass
            a ``str``, as follows::

                >>> print_list('zip://projects.zip')

            Or, an filesystem instance::

                >>> from fs.osfs import OSFS
                >>> projects_fs = OSFS('~/')
                >>> print_list(projects_fs)

        """
        ...

registry = ...
