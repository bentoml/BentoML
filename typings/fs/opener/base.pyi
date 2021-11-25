

import abc
import typing
from typing import List, Text

import six

from ..base import FS
from .parse import ParseResult

"""`Opener` abstract base class.
"""
if typing.TYPE_CHECKING: ...

@six.add_metaclass(abc.ABCMeta)
class Opener:
    """The base class for filesystem openers.

    An opener is responsible for opening a filesystem for a given
    protocol.

    """

    protocols: List[Text] = ...
    def __repr__(self) -> Text: ...
    @abc.abstractmethod
    def open_fs(
        self,
        fs_url: Text,
        parse_result: ParseResult,
        writeable: bool,
        create: bool,
        cwd: Text,
    ) -> FS:
        """Open a filesystem object from a FS URL.

        Arguments:
            fs_url (str): A filesystem URL.
            parse_result (~fs.opener.parse.ParseResult): A parsed
                filesystem URL.
            writeable (bool): `True` if the filesystem must be writable.
            create (bool): `True` if the filesystem should be created
                if it does not exist.
            cwd (str): The current working directory (generally only
                relevant for OS filesystems).

        Raises:
            fs.opener.errors.OpenerError: If a filesystem could not
                be opened for any reason.

        Returns:
            `~fs.base.FS`: A filesystem instance.

        """
        ...
