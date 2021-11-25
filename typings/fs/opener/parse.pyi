

import collections
import typing
from typing import Text

"""Function to parse FS URLs in to their constituent parts.
"""
if typing.TYPE_CHECKING: ...

class ParseResult(
    collections.namedtuple(
        "ParseResult",
        ["protocol", "username", "password", "resource", "params", "path"],
    )
):
    """A named tuple containing fields of a parsed FS URL.

    Attributes:
        protocol (str): The protocol part of the url, e.g. ``osfs``
            or ``ftp``.
        username (str, optional): A username, or `None`.
        password (str, optional): A password, or `None`.
        resource (str): A *resource*, typically a domain and path, e.g.
            ``ftp.example.org/dir``.
        params (dict): A dictionary of parameters extracted from the
            query string.
        path (str, optional): A path within the filesystem, or `None`.

    """

    ...

_RE_FS_URL = ...

def parse_fs_url(fs_url: Text) -> ParseResult:
    """Parse a Filesystem URL and return a `ParseResult`.

    Arguments:
        fs_url (str): A filesystem URL.

    Returns:
        ~fs.opener.parse.ParseResult: a parse result instance.

    Raises:
        ~fs.errors.ParseError: if the FS URL is not valid.

    """
    ...
