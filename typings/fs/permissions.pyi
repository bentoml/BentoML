

import typing
from typing import Iterable, Iterator, List, Optional, Tuple, Type, Union

import six

from ._typing import Text

"""Abstract permissions container.
"""
if typing.TYPE_CHECKING: ...

def make_mode(init: Union[int, Iterable[Text], None]) -> int:
    """Make a mode integer from an initial value."""
    ...

class _PermProperty:
    """Creates simple properties to get/set permissions."""

    def __init__(self, name: Text) -> None: ...
    def __get__(
        self, obj: Permissions, obj_type: Optional[Type[Permissions]] = ...
    ) -> bool: ...
    def __set__(self, obj: Permissions, value: bool) -> None: ...

@six.python_2_unicode_compatible
class Permissions:
    """An abstraction for file system permissions.

    Permissions objects store information regarding the permissions
    on a resource. It supports Linux permissions, but is generic enough
    to manage permission information from almost any filesystem.

    Example:
        >>> from fs.permissions import Permissions
        >>> p = Permissions(user='rwx', group='rw-', other='r--')
        >>> print(p)
        rwxrw-r--
        >>> p.mode
        500
        >>> oct(p.mode)
        '0o764'

    """

    _LINUX_PERMS: List[Tuple[Text, int]] = ...
    _LINUX_PERMS_NAMES: List[Text] = ...
    def __init__(
        self,
        names: Optional[Iterable[Text]] = ...,
        mode: Optional[int] = ...,
        user: Optional[Text] = ...,
        group: Optional[Text] = ...,
        other: Optional[Text] = ...,
        sticky: Optional[bool] = ...,
        setuid: Optional[bool] = ...,
        setguid: Optional[bool] = ...,
    ) -> None:
        """Create a new `Permissions` instance.

        Arguments:
            names (list, optional): A list of permissions.
            mode (int, optional): A mode integer.
            user (str, optional): A triplet of *user* permissions, e.g.
                ``"rwx"`` or ``"r--"``
            group (str, optional): A triplet of *group* permissions, e.g.
                ``"rwx"`` or ``"r--"``
            other (str, optional): A triplet of *other* permissions, e.g.
                ``"rwx"`` or ``"r--"``
            sticky (bool, optional): A boolean for the *sticky* bit.
            setuid (bool, optional): A boolean for the *setuid* bit.
            setguid (bool, optional): A boolean for the *setguid* bit.

        """
        ...
    def __repr__(self) -> Text: ...
    def __str__(self) -> Text: ...
    def __iter__(self) -> Iterator[Text]: ...
    def __contains__(self, permission: object) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    @classmethod
    def parse(cls, ls: Text) -> Permissions:
        """Parse permissions in Linux notation."""
        ...
    @classmethod
    def load(cls, permissions: List[Text]) -> Permissions:
        """Load a serialized permissions object."""
        ...
    @classmethod
    def create(cls, init: Union[int, Iterable[Text], None] = ...) -> Permissions:
        """Create a permissions object from an initial value.

        Arguments:
            init (int or list, optional): May be None to use `0o777`
                permissions, a mode integer, or a list of permission names.

        Returns:
            int: mode integer that may be used for instance by `os.makedir`.

        Example:
            >>> Permissions.create(None)
            Permissions(user='rwx', group='rwx', other='rwx')
            >>> Permissions.create(0o700)
            Permissions(user='rwx', group='', other='')
            >>> Permissions.create(['u_r', 'u_w', 'u_x'])
            Permissions(user='rwx', group='', other='')

        """
        ...
    @classmethod
    def get_mode(cls, init: Union[int, Iterable[Text], None]) -> int:
        """Convert an initial value to a mode integer."""
        ...
    def copy(self) -> Permissions:
        """Make a copy of this permissions object."""
        ...
    def dump(self) -> List[Text]:
        """Get a list suitable for serialization."""
        ...
    def as_str(self) -> Text:
        """Get a Linux-style string representation of permissions."""
        ...
    @property
    def mode(self) -> int:
        """`int`: mode integer."""
        ...
    @mode.setter
    def mode(self, mode: int) -> None: ...
    u_r = ...
    u_w = ...
    u_x = ...
    g_r = ...
    g_w = ...
    g_x = ...
    o_r = ...
    o_w = ...
    o_x = ...
    sticky = ...
    setuid = ...
    setguid = ...
    def add(self, *permissions: Text) -> None:
        """Add permission(s).

        Arguments:
            *permissions (str): Permission name(s), such as ``'u_w'``
                or ``'u_x'``.

        """
        ...
    def remove(self, *permissions: Text) -> None:
        """Remove permission(s).

        Arguments:
            *permissions (str): Permission name(s), such as ``'u_w'``
                or ``'u_x'``.s

        """
        ...
    def check(self, *permissions: Text) -> bool:
        """Check if one or more permissions are enabled.

        Arguments:
            *permissions (str): Permission name(s), such as ``'u_w'``
                or ``'u_x'``.

        Returns:
            bool: `True` if all given permissions are set.

        """
        ...
