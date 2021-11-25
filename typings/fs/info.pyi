

import typing
from datetime import datetime
from typing import Any, Callable, List, Mapping, Optional, Union

import six

from ._typing import Text, overload
from .enums import ResourceType
from .permissions import Permissions

"""Container for filesystem resource informations.
"""
RawInfo = Mapping[Text, Mapping[Text, object]]
ToDatetime = Callable[[int], datetime]
T = typing.TypeVar("T")

@six.python_2_unicode_compatible
class Info:
    """Container for :ref:`info`.

    Resource information is returned by the following methods:

         * `~fs.base.FS.getinfo`
         * `~fs.base.FS.scandir`
         * `~fs.base.FS.filterdir`

    Arguments:
        raw_info (dict): A dict containing resource info.
        to_datetime (callable): A callable that converts an
            epoch time to a datetime object. The default uses
            `~fs.time.epoch_to_datetime`.

    """

    __slots__ = ["raw", "_to_datetime", "namespaces"]
    def __init__(self, raw_info: RawInfo, to_datetime: ToDatetime = ...) -> None:
        """Create a resource info object from a raw info dict."""
        ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    @overload
    def get(self, namespace: Text, key: Text) -> Any: ...
    @overload
    def get(self, namespace: Text, key: Text, default: T) -> Union[Any, T]: ...
    def get(
        self, namespace: Text, key: Text, default: Optional[Any] = ...
    ) -> Optional[Any]:
        """Get a raw info value.

        Arguments:
            namespace (str): A namespace identifier.
            key (str): A key within the namespace.
            default (object, optional): A default value to return
                if either the namespace or the key within the namespace
                is not found.

        Example:
            >>> info = my_fs.getinfo("foo.py", namespaces=["details"])
            >>> info.get('details', 'type')
            2

        """
        ...
    def is_writeable(self, namespace: Text, key: Text) -> bool:
        """Check if a given key in a namespace is writable.

        When creating an `Info` object, you can add a ``_write`` key to
        each raw namespace that lists which keys are writable or not.

        Arguments:
            namespace (str): A namespace identifier.
            key (str): A key within the namespace.

        Returns:
            bool: `True` if the key can be modified, `False` otherwise.

        Example:
            Create an `Info` object that marks only the ``modified`` key
            as writable in the ``details`` namespace::

                >>> now = time.time()
                >>> info = Info({
                ...     "basic": {"name": "foo", "is_dir": False},
                ...     "details": {
                ...         "modified": now,
                ...         "created": now,
                ...         "_write": ["modified"],
                ...     }
                ... })
                >>> info.is_writeable("details", "created")
                False
                >>> info.is_writeable("details", "modified")
                True

        """
        ...
    def has_namespace(self, namespace: Text) -> bool:
        """Check if the resource info contains a given namespace.

        Arguments:
            namespace (str): A namespace identifier.

        Returns:
            bool: `True` if the namespace was found, `False` otherwise.

        """
        ...
    def copy(self, to_datetime: Optional[ToDatetime] = ...) -> Info:
        """Create a copy of this resource info object."""
        ...
    def make_path(self, dir_path: Text) -> Text:
        """Make a path by joining ``dir_path`` with the resource name.

        Arguments:
            dir_path (str): A path to a directory.

        Returns:
            str: A path to the resource.

        """
        ...
    @property
    def name(self) -> Text:
        """`str`: the resource name."""
        ...
    @property
    def suffix(self) -> Text:
        """`str`: the last component of the name (with dot).

        In case there is no suffix, an empty string is returned.

        Example:
            >>> info = my_fs.getinfo("foo.py")
            >>> info.suffix
            '.py'
            >>> info2 = my_fs.getinfo("bar")
            >>> info2.suffix
            ''

        """
        ...
    @property
    def suffixes(self) -> List[Text]:
        """`List`: a list of any suffixes in the name.

        Example:
            >>> info = my_fs.getinfo("foo.tar.gz")
            >>> info.suffixes
            ['.tar', '.gz']

        """
        ...
    @property
    def stem(self) -> Text:
        """`str`: the name minus any suffixes.

        Example:
            >>> info = my_fs.getinfo("foo.tar.gz")
            >>> info.stem
            'foo'

        """
        ...
    @property
    def is_dir(self) -> bool:
        """`bool`: `True` if the resource references a directory."""
        ...
    @property
    def is_file(self) -> bool:
        """`bool`: `True` if the resource references a file."""
        ...
    @property
    def is_link(self) -> bool:
        """`bool`: `True` if the resource is a symlink."""
        ...
    @property
    def type(self) -> ResourceType:
        """`~fs.enums.ResourceType`: the type of the resource.

        Requires the ``"details"`` namespace.

        Raises:
            ~fs.errors.MissingInfoNamespace: if the 'details'
                namespace is not in the Info.

        """
        ...
    @property
    def accessed(self) -> Optional[datetime]:
        """`~datetime.datetime`: the resource last access time, or `None`.

        Requires the ``"details"`` namespace.

        Raises:
            ~fs.errors.MissingInfoNamespace: if the ``"details"``
                namespace is not in the Info.

        """
        ...
    @property
    def modified(self) -> Optional[datetime]:
        """`~datetime.datetime`: the resource last modification time, or `None`.

        Requires the ``"details"`` namespace.

        Raises:
            ~fs.errors.MissingInfoNamespace: if the ``"details"``
                namespace is not in the Info.

        """
        ...
    @property
    def created(self) -> Optional[datetime]:
        """`~datetime.datetime`: the resource creation time, or `None`.

        Requires the ``"details"`` namespace.

        Raises:
            ~fs.errors.MissingInfoNamespace: if the ``"details"``
                namespace is not in the Info.

        """
        ...
    @property
    def metadata_changed(self) -> Optional[datetime]:
        """`~datetime.datetime`: the resource metadata change time, or `None`.

        Requires the ``"details"`` namespace.

        Raises:
            ~fs.errors.MissingInfoNamespace: if the ``"details"``
                namespace is not in the Info.

        """
        ...
    @property
    def permissions(self) -> Optional[Permissions]:
        """`Permissions`: the permissions of the resource, or `None`.

        Requires the ``"access"`` namespace.

        Raises:
            ~fs.errors.MissingInfoNamespace: if the ``"access"``
                namespace is not in the Info.

        """
        ...
    @property
    def size(self) -> int:
        """`int`: the size of the resource, in bytes.

        Requires the ``"details"`` namespace.

        Raises:
            ~fs.errors.MissingInfoNamespace: if the ``"details"``
                namespace is not in the Info.

        """
        ...
    @property
    def user(self) -> Optional[Text]:
        """`str`: the owner of the resource, or `None`.

        Requires the ``"access"`` namespace.

        Raises:
            ~fs.errors.MissingInfoNamespace: if the ``"access"``
                namespace is not in the Info.

        """
        ...
    @property
    def uid(self) -> Optional[int]:
        """`int`: the user id of the resource, or `None`.

        Requires the ``"access"`` namespace.

        Raises:
            ~fs.errors.MissingInfoNamespace: if the ``"access"``
                namespace is not in the Info.

        """
        ...
    @property
    def group(self) -> Optional[Text]:
        """`str`: the group of the resource owner, or `None`.

        Requires the ``"access"`` namespace.

        Raises:
            ~fs.errors.MissingInfoNamespace: if the ``"access"``
                namespace is not in the Info.

        """
        ...
    @property
    def gid(self) -> Optional[int]:
        """`int`: the group id of the resource, or `None`.

        Requires the ``"access"`` namespace.

        Raises:
            ~fs.errors.MissingInfoNamespace: if the ``"access"``
                namespace is not in the Info.

        """
        ...
    @property
    def target(self) -> Optional[Text]:
        """`str`: the link target (if resource is a symlink), or `None`.

        Requires the ``"link"`` namespace.

        Raises:
            ~fs.errors.MissingInfoNamespace: if the ``"link"``
                namespace is not in the Info.

        """
        ...
