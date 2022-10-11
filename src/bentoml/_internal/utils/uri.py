import os
import pathlib
from urllib.parse import unquote
from urllib.parse import urlparse
from urllib.request import url2pathname

import psutil


def path_to_uri(path: str) -> str:
    """
    Convert a path to a URI.

    Args:
        path: Path to convert to URI.

    Returns:
        URI string. (quoted, absolute)
    """
    path = os.path.abspath(path)
    if psutil.WINDOWS:
        return pathlib.PureWindowsPath(path).as_uri()
    if psutil.POSIX:
        return pathlib.PurePosixPath(path).as_uri()
    raise ValueError("Unsupported OS")


def uri_to_path(uri: str) -> str:
    """
    Convert a file URI to a path.

    Args:
        uri: URI to convert to path.

    Returns:
        Path string. (unquoted)
    """
    parsed = urlparse(uri)
    if parsed.scheme not in ("file", "filesystem", "unix"):
        raise ValueError("Unsupported URI scheme")
    host = "{0}{0}{mnt}{0}".format(os.path.sep, mnt=parsed.netloc)
    return os.path.normpath(os.path.join(host, url2pathname(unquote(parsed.path))))
