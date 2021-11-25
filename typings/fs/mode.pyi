

import typing
from typing import FrozenSet, Set, Union

import six

from ._typing import Text

"""Abstract I/O mode container.

Mode strings are used in in `~fs.base.FS.open` and
`~fs.base.FS.openbin`.

"""
if typing.TYPE_CHECKING: ...
__all__ = ["Mode", "check_readable", "check_writable", "validate_openbin_mode"]

@six.python_2_unicode_compatible
class Mode(typing.Container[Text]):
    """An abstraction for I/O modes.

    A mode object provides properties that can be used to interrogate the
    `mode strings <https://docs.python.org/3/library/functions.html#open>`_
    used when opening files.

    Example:
        >>> mode = Mode('rb')
        >>> mode.reading
        True
        >>> mode.writing
        False
        >>> mode.binary
        True
        >>> mode.text
        False

    """

    def __init__(self, mode: Text) -> None:
        """Create a new `Mode` instance.

        Arguments:
            mode (str): A *mode* string, as used by `io.open`.

        Raises:
            ValueError: If the mode string is invalid.

        """
        ...
    def __repr__(self) -> Text: ...
    def __str__(self) -> Text: ...
    def __contains__(self, character: object) -> bool:
        """Check if a mode contains a given character."""
        ...
    def to_platform(self) -> Text:
        """Get a mode string for the current platform.

        Currently, this just removes the 'x' on PY2 because PY2 doesn't
        support exclusive mode.

        """
        ...
    def to_platform_bin(self) -> Text:
        """Get a *binary* mode string for the current platform.

        This removes the 't' and adds a 'b' if needed.

        """
        ...
    def validate(self, _valid_chars: Union[Set[Text], FrozenSet[Text]] = ...) -> None:
        """Validate the mode string.

        Raises:
            ValueError: if the mode contains invalid chars.

        """
        ...
    def validate_bin(self) -> None:
        """Validate a mode for opening a binary file.

        Raises:
            ValueError: if the mode contains invalid chars.

        """
        ...
    @property
    def create(self) -> bool:
        """`bool`: `True` if the mode would create a file."""
        ...
    @property
    def reading(self) -> bool:
        """`bool`: `True` if the mode permits reading."""
        ...
    @property
    def writing(self) -> bool:
        """`bool`: `True` if the mode permits writing."""
        ...
    @property
    def appending(self) -> bool:
        """`bool`: `True` if the mode permits appending."""
        ...
    @property
    def updating(self) -> bool:
        """`bool`: `True` if the mode permits both reading and writing."""
        ...
    @property
    def truncate(self) -> bool:
        """`bool`: `True` if the mode would truncate an existing file."""
        ...
    @property
    def exclusive(self) -> bool:
        """`bool`: `True` if the mode require exclusive creation."""
        ...
    @property
    def binary(self) -> bool:
        """`bool`: `True` if a mode specifies binary."""
        ...
    @property
    def text(self) -> bool:
        """`bool`: `True` if a mode specifies text."""
        ...

def check_readable(mode: Text) -> bool:
    """Check a mode string allows reading.

    Arguments:
        mode (str): A mode string, e.g. ``"rt"``

    Returns:
        bool: `True` if the mode allows reading.

    """
    ...

def check_writable(mode: Text) -> bool:
    """Check a mode string allows writing.

    Arguments:
        mode (str): A mode string, e.g. ``"wt"``

    Returns:
        bool: `True` if the mode allows writing.

    """
    ...

def validate_open_mode(mode: Text) -> None:
    """Check ``mode`` parameter of `~fs.base.FS.open` is valid.

    Arguments:
        mode (str): Mode parameter.

    Raises:
        `ValueError` if mode is not valid.

    """
    ...

def validate_openbin_mode(
    mode: Text, _valid_chars: Union[Set[Text], FrozenSet[Text]] = ...
) -> None:
    """Check ``mode`` parameter of `~fs.base.FS.openbin` is valid.

    Arguments:
        mode (str): Mode parameter.

    Raises:
        `ValueError` if mode is not valid.

    """
    ...
